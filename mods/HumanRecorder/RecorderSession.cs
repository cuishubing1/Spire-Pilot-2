using System.Reflection;
using System.Security.Cryptography;
using System.Diagnostics;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace Sts2HumanRecorder;

internal static class RecorderSession
{
    private const string ExpectedAssemblySha256 = "a1f9e653f1e28e4076558fee1e60d218619cb7e057b887c6417f62c62c6d7a52";
    private static readonly object Gate = new();
    private static readonly object HealthGate = new();
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };
    private static readonly string ProcessSessionId = "session-" + Guid.NewGuid().ToString("N");
    private static readonly List<DecisionAnchor> Anchors = new();
    private static AsyncRecordWriter? _writer;
    private static string? _partialPath;
    private static string? _runId;
    private static string _previousHash = new('0', 64);
    private static long _sequence;
    private static long? _lastDecisionSequence;
    private static string? _actorId;
    private static int _attemptId;
    private static int _rollbackCount;
    private static bool _pendingResume;
    private static int _pendingFromAttempt;
    private static long? _pendingFromDecision;
    private static bool _recordingDisabled;
    private static string? _lastError;
    private static long _healthGeneration;
    private static long _lastHealthWritten;
    private static readonly Lazy<string> DefaultInboxPath = new(ResolveDefaultInboxCore, true);
    private static string _storageMode = "unresolved";
    private static string? _storageFallbackReason;
    private static PendingActionOutcome? _pendingActionOutcome;
    private static int _warningCount;
    private static string? _gameAssemblySha256;
    private static string? _gameAssemblyVersion;

    private sealed record PendingActionOutcome(
        long SourceDecisionSequence,
        int AttemptId,
        string OriginPhase,
        string? SourceId,
        string? SourceInstanceId,
        string? OriginEventId,
        string? OriginRoomType,
        object? OriginFloor,
        object? OriginMapCoord);

    public static void Initialize()
    {
        lock (Gate)
        {
            _actorId = LoadActorId();
            ContentProvenance.PrewarmEnvironmentSummary();
            WriteHealthUnsafe("ready");
        }
        AppDomain.CurrentDomain.ProcessExit += (_, _) => Suspend("process_exit");
    }

    public static void ReportFatalInitialization(Exception ex)
    {
        lock (Gate)
        {
            _recordingDisabled = true;
            _lastError = ex.GetType().Name + ": " + ex.Message;
            try { WriteHealthUnsafe("disabled"); } catch { }
        }
    }

    internal static RecorderUiState GetUiState()
    {
        lock (Gate)
        {
            var status = _recordingDisabled ? "error"
                : _pendingResume ? "restoring"
                : _writer is null ? "idle"
                : _warningCount > 0 ? "warning"
                : "recording";
            return new RecorderUiState(status, _warningCount);
        }
    }

    public static void AssertCompatibleGame()
    {
        var assembly = typeof(MegaCrit.Sts2.Core.Modding.ModInitializerAttribute).Assembly;
        var actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(assembly.Location))).ToLowerInvariant();
        _gameAssemblySha256 = actual;
        _gameAssemblyVersion = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? assembly.GetName().Version?.ToString();
        if (!string.Equals(actual, ExpectedAssemblySha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"HumanRecorder is locked to STS2 v0.107.1/build 23811903; sts2.dll was {actual}");
    }

    public static void StartNewRun()
    {
        lock (Gate)
        {
            if (_writer is null && File.Exists(ActivePath())) TryOpenActiveUnsafe();
            if (_writer is not null) EndRunUnsafe("superseded_by_new_run", null);
            StartRunUnsafe("new_run");
        }
    }

    public static void ResumeRun()
    {
        lock (Gate)
        {
            var reopened = false;
            if (_writer is null) reopened = TryOpenActiveUnsafe();
            if (_writer is null) StartRunUnsafe("saved_run_late_attach");

            // SL is an infrequent durability boundary. Commit the previous attempt
            // before announcing the resumed branch.
            if (_writer is not null && !_writer.Flush(true))
                DisableUnsafe("writer could not durably flush before resume", _writer.Fault);
            if (_recordingDisabled) return;

            _pendingFromAttempt = _attemptId;
            _pendingFromDecision = _lastDecisionSequence;
            _pendingActionOutcome = null;
            _attemptId++;
            _pendingResume = true;
            WriteUnsafe("resume", new Dictionary<string, object?>
            {
                ["run_id"] = _runId, ["session_id"] = ProcessSessionId,
                ["from_attempt_id"] = _pendingFromAttempt, ["to_attempt_id"] = _attemptId,
                ["reopened_after_process_exit"] = reopened,
                ["from_decision_sequence"] = _pendingFromDecision
            });
        }
    }

    public static ActionCommitToken? RecordAction(string phase, string actionId, object? instance, object?[] args)
    {
        lock (Gate)
        {
            if (_recordingDisabled) return null;
            if (_writer is null && !TryOpenActiveUnsafe()) StartRunUnsafe("late_attach");
            if (_recordingDisabled || _writer is null) return null;
            var started = Stopwatch.GetTimestamp();
            var obs = StateExporter.Capture(phase);
            StateExporter.EnrichFromActionContext(obs, phase, actionId, instance, args);
            ResolvePendingActionOutcomeUnsafe(obs);
            var observationDone = Stopwatch.GetTimestamp();
            var auditState = StateExporter.CaptureAuditState(phase);
            var auditDone = Stopwatch.GetTimestamp();
            if (!string.Equals(auditState.GetValueOrDefault("capture_quality")?.ToString(), "complete", StringComparison.Ordinal))
            {
                obs["capture_quality"] = "partial";
                var errors = new List<string> { "native_audit_state_incomplete" };
                errors.AddRange(ReflectionUtil.Items(auditState.GetValueOrDefault("capture_errors")).Select(x => x.ToString() ?? "unknown"));
                obs["capture_errors"] = errors;
            }
            var action = ActionEncoder.Encode(actionId, instance, args);
            ReconcileAction(obs, action);
            var contentScope = ContentProvenance.ClassifyDecision(obs, action);
            var encodeDone = Stopwatch.GetTimestamp();

            if (_pendingResume) ResolveResumeUnsafe(obs);
            var quality = obs.TryGetValue("capture_quality", out var q) ? q : "partial";
            if (!string.Equals(quality?.ToString(), "complete", StringComparison.Ordinal)) _warningCount++;
            var decisionSequence = _sequence;
            var telemetry = new Dictionary<string, object?>
            {
                ["observation_ms"] = Math.Round(Stopwatch.GetElapsedTime(started, observationDone).TotalMilliseconds, 3),
                ["audit_ms"] = Math.Round(Stopwatch.GetElapsedTime(observationDone, auditDone).TotalMilliseconds, 3),
                ["encode_and_classify_ms"] = Math.Round(Stopwatch.GetElapsedTime(auditDone, encodeDone).TotalMilliseconds, 3),
                ["writer_queue_depth"] = _writer.Metrics().GetValueOrDefault("queue_depth")
            };
            var fingerprintStarted = Stopwatch.GetTimestamp();
            var anchor = StateFingerprint.Build(obs, decisionSequence, _attemptId);
            telemetry["fingerprint_ms"] = Math.Round(Stopwatch.GetElapsedTime(fingerprintStarted).TotalMilliseconds, 3);
            telemetry["capture_ms"] = Math.Round(Stopwatch.GetElapsedTime(started).TotalMilliseconds, 3);
            var decisionPayload = new Dictionary<string, object?>
            {
                ["run_id"] = _runId, ["step_id"] = decisionSequence, ["attempt_id"] = _attemptId,
                ["phase"] = phase, ["observation"] = obs, ["action"] = action,
                ["audit_state"] = auditState,
                ["capture_quality"] = quality, ["content_scope"] = contentScope,
                ["policy_id"] = "human_v4",
                ["commit_status"] = "invoked",
                ["telemetry"] = telemetry
            };
            var commit = new ActionCommitToken(decisionPayload);
            WriteUnsafe("decision", decisionPayload, commit);
            if (_recordingDisabled) return null;
            if (actionId == "use_potion" && phase != "combat_play")
                ArmPendingActionOutcomeUnsafe(decisionSequence, phase, obs, action);
            Anchors.Add(anchor);
            _lastDecisionSequence = decisionSequence;
            if (decisionSequence % 64 == 0) QueueHealthUpdateUnsafe("recording");
            return commit;
        }
    }

    public static void RecordEngineEvent(string eventType, Dictionary<string, object?> details)
    {
        lock (Gate)
        {
            // Engine events are audit-only supplements. Never create a run solely because
            // background engine code fired outside an active recorded climb.
            if (_writer is null || _recordingDisabled) return;
            WriteUnsafe("engine_event", new Dictionary<string, object?>
            {
                ["run_id"] = _runId, ["attempt_id"] = _attemptId,
                ["after_decision_sequence"] = _lastDecisionSequence,
                ["event_type"] = eventType, ["details"] = details
            });
        }
    }

    public static void EndRun(string reason, bool? won)
    {
        lock (Gate)
        {
            if (_writer is null) TryOpenActiveUnsafe();
            EndRunUnsafe(reason, won);
        }
    }

    private static void ResolveResumeUnsafe(Dictionary<string, object?> observation)
    {
        var current = StateFingerprint.Build(observation, _sequence, _attemptId);
        var match = StateFingerprint.Match(current, Anchors, _pendingFromAttempt);
        _rollbackCount++;
        var recordType = match.Anchor is null || match.Quality is "unmatched" or "location_only"
            ? "resume_unmatched" : "rollback";
        if (recordType == "resume_unmatched") _warningCount++;
        WriteUnsafe(recordType, new Dictionary<string, object?>
        {
            ["run_id"] = _runId, ["rollback_id"] = _rollbackCount,
            ["from_attempt_id"] = _pendingFromAttempt, ["to_attempt_id"] = _attemptId,
            ["from_decision_sequence"] = _pendingFromDecision,
            ["rollback_target_sequence"] = match.Anchor?.Sequence,
            ["rollback_target_attempt_id"] = match.Anchor?.AttemptId,
            ["discarded_decision_range"] = match.Anchor is null || _pendingFromDecision is null
                ? null : new[] { match.Anchor.Sequence, _pendingFromDecision.Value },
            ["match_quality"] = match.Quality, ["match_confidence"] = match.Confidence,
            ["match_reason"] = match.Reason, ["restored_state_hash"] = current.ExactHash,
            ["restored_semantic_hash"] = current.SemanticHash, ["room_key"] = current.RoomKey,
            ["canonical_boundary"] = recordType == "rollback" ? "resolved" : "quarantine"
        });
        _pendingResume = false;
        _pendingFromDecision = null;
    }

    private static void StartRunUnsafe(string source)
    {
        _recordingDisabled = false;
        _lastError = null;
        CardIdentity.ResetForRun();
        _runId = $"human-{DateTime.UtcNow:yyyyMMddTHHmmssfffZ}-{Guid.NewGuid():N}";
        _sequence = 0; _previousHash = new string('0', 64); _attemptId = 0; _rollbackCount = 0;
        _lastDecisionSequence = null; _pendingResume = false; _pendingActionOutcome = null; _warningCount = 0; Anchors.Clear();
        var inbox = ResolveInbox();
        Directory.CreateDirectory(inbox);
        _partialPath = Path.Combine(inbox, _runId + ".jsonl.partial");
        OpenWriterUnsafe(FileMode.CreateNew);
        SaveActiveUnsafe();
        WriteUnsafe("recorder_start", new Dictionary<string, object?>
        {
            ["schema_version"] = MainFile.LiveSchemaVersion, ["recorder_version"] = MainFile.RecorderVersion,
            ["actor_id"] = _actorId, ["source"] = source, ["game"] = GameFingerprint(),
            ["environment"] = ContentProvenance.EnvironmentSummary(),
            ["storage"] = new
            {
                mode = _storageMode,
                root = Directory.GetParent(ResolveInbox())?.FullName ?? ResolveInbox(),
                fallback_reason = _storageFallbackReason
            },
            ["hook_manifest"] = PatchRegistry.HookManifest(),
            ["privacy"] = new { contains_account_id = false, contains_input_events = false }
        });
        WriteUnsafe("run_start", new Dictionary<string, object?>
        {
            ["run_id"] = _runId, ["session_id"] = ProcessSessionId, ["attempt_id"] = 0,
            ["run_context"] = StateExporter.CaptureRunContext(),
            ["observation"] = StateExporter.Capture("bundle_select"),
            ["audit_state"] = StateExporter.CaptureAuditState("bundle_select")
        });
        WriteHealthUnsafe("recording");
    }

    private static bool TryOpenActiveUnsafe()
    {
        var activePath = ActivePath();
        if (!File.Exists(activePath)) return false;
        try
        {
            using var active = JsonDocument.Parse(File.ReadAllText(activePath, Encoding.UTF8));
            var path = active.RootElement.GetProperty("partial_path").GetString();
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return false;
            ResetMemoryUnsafe();
            _partialPath = Path.GetFullPath(path);
            LoadAndVerifyPrefixUnsafe(_partialPath);
            OpenWriterUnsafe(FileMode.Append);
            WriteUnsafe("session_resume", new Dictionary<string, object?>
            {
                ["run_id"] = _runId, ["session_id"] = ProcessSessionId,
                ["attempt_id"] = _attemptId, ["records_before_resume"] = _sequence
            });
            return true;
        }
        catch (Exception ex)
        {
            MainFile.Logger.Error($"Active HumanRecorder run could not be reopened and was left untouched: {ex}");
            ResetMemoryUnsafe();
            return false;
        }
    }

    private static void LoadAndVerifyPrefixUnsafe(string path)
    {
        var expectedSequence = 0L;
        var previous = new string('0', 64);
        foreach (var line in File.ReadLines(path, Encoding.UTF8))
        {
            var record = JsonSerializer.Deserialize<SortedDictionary<string, JsonElement>>(line, JsonOptions)
                ?? throw new InvalidDataException("null JSONL record");
            if (!record.TryGetValue("record_sha256", out var hashElement)) throw new InvalidDataException("record hash missing");
            var storedHash = hashElement.GetString() ?? "";
            record.Remove("record_sha256");
            var canonical = JsonSerializer.Serialize(record, JsonOptions);
            var actualHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical))).ToLowerInvariant();
            if (actualHash != storedHash) throw new InvalidDataException($"record hash mismatch at sequence {expectedSequence}");
            if (record["sequence"].GetInt64() != expectedSequence) throw new InvalidDataException("record sequence gap");
            if (record["prev_record_sha256"].GetString() != previous) throw new InvalidDataException("record hash chain broken");
            var type = record["record_type"].GetString();
            var payload = record["payload"];
            if (type == "recorder_start")
            {
                var schema = payload.GetProperty("schema_version").GetString();
                if (!string.Equals(schema, MainFile.LiveSchemaVersion, StringComparison.Ordinal))
                    throw new InvalidDataException($"active recording schema {schema} cannot be appended by {MainFile.LiveSchemaVersion}");
            }
            if (type == "run_start") _runId = payload.GetProperty("run_id").GetString();
            if (type == "decision")
            {
                var attempt = payload.TryGetProperty("attempt_id", out var attemptElement) ? attemptElement.GetInt32() : 0;
                _attemptId = Math.Max(_attemptId, attempt);
                _lastDecisionSequence = expectedSequence;
                Anchors.Add(StateFingerprint.Build(payload.GetProperty("observation"), expectedSequence, attempt));
                if (payload.TryGetProperty("capture_quality", out var qualityElement)
                    && qualityElement.GetString() != "complete") _warningCount++;
            }
            if (type is "rollback" or "resume_unmatched")
            {
                _rollbackCount++;
                if (type == "resume_unmatched") _warningCount++;
            }
            previous = storedHash; expectedSequence++;
        }
        if (expectedSequence < 2 || string.IsNullOrWhiteSpace(_runId)) throw new InvalidDataException("active run prefix is incomplete");
        _sequence = expectedSequence; _previousHash = previous;
    }

    private static void Suspend(string reason)
    {
        lock (Gate)
        {
            if (_writer is null) return;
            try
            {
                WriteUnsafe("session_end", new Dictionary<string, object?>
                {
                    ["run_id"] = _runId, ["session_id"] = ProcessSessionId,
                    ["reason"] = reason, ["attempt_id"] = _attemptId
                });
            }
            catch { }
            CloseWriterUnsafe();
        }
    }

    private static void EndRunUnsafe(string reason, bool? won)
    {
        if (_writer is null) return;
        try
        {
            var finalObservation = StateExporter.Capture("game_over");
            won ??= InferVictory(reason, finalObservation);
            WriteUnsafe("run_end", new Dictionary<string, object?>
            {
                ["run_id"] = _runId, ["reason"] = reason, ["won"] = won,
                ["attempt_id"] = _attemptId, ["rollback_count"] = _rollbackCount,
                ["writer_metrics"] = _writer?.Metrics(),
                ["observation"] = finalObservation,
                ["audit_state"] = StateExporter.CaptureAuditState("game_over")
            });
        }
        finally
        {
            CloseWriterUnsafe();
            if (!_recordingDisabled && _partialPath is not null && File.Exists(_partialPath))
                File.Move(_partialPath, _partialPath.Replace(".jsonl.partial", ".jsonl"), false);
            if (!_recordingDisabled) DeleteActiveUnsafe();
            ResetMemoryUnsafe();
            WriteHealthUnsafe(_recordingDisabled ? "disabled" : "ready");
        }
    }

    private static bool? InferVictory(string reason, Dictionary<string, object?> observation)
    {
        if (reason == "abandoned") return false;
        if (reason != "game_ended") return null;
        var player = observation.GetValueOrDefault("player") as Dictionary<string, object?>;
        var hp = ReflectionUtil.Int(player?.GetValueOrDefault("hp"));
        if (hp <= 0) return false;
        // RunManager.OnEnded is the terminal signal. Abandon has its own hook, so a
        // terminal run with positive HP is a victory even if IsGameOver updates later.
        return true;
    }

    private static void OpenWriterUnsafe(FileMode mode)
    {
        _writer = new AsyncRecordWriter(_partialPath!, mode, _previousHash);
    }

    private static void CloseWriterUnsafe()
    {
        var writer = _writer;
        if (writer is null) return;
        try
        {
            writer.Flush(true);
            writer.Dispose();
            _previousHash = writer.PreviousHash;
            if (writer.Fault is not null) DisableUnsafe("background writer failed", writer.Fault);
        }
        catch (Exception ex) { DisableUnsafe("background writer close failed", ex); }
        finally { _writer = null; }
    }

    private static void WriteUnsafe(string recordType, Dictionary<string, object?> payload, ActionCommitToken? commit = null)
    {
        if (_writer is null || _recordingDisabled) return;
        var sequence = _sequence;
        var record = new SortedDictionary<string, object?>(StringComparer.Ordinal)
        {
            ["record_type"] = recordType, ["sequence"] = sequence, ["timestamp_utc"] = DateTime.UtcNow.ToString("O"),
            ["payload"] = payload
        };
        if (_writer.TryEnqueue(record, commit))
        {
            _sequence++;
            return;
        }
        DisableUnsafe("background writer queue rejected a record", _writer.Fault);
    }

    private static void SaveActiveUnsafe()
    {
        if (_partialPath is null || _runId is null) return;
        var path = ActivePath();
        var temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        var value = new SortedDictionary<string, object?>(StringComparer.Ordinal)
        {
            ["schema_version"] = "human-active-0.2.0", ["run_id"] = _runId,
            ["partial_path"] = _partialPath, ["updated_at"] = DateTime.UtcNow.ToString("O")
        };
        File.WriteAllText(temp, JsonSerializer.Serialize(value, JsonOptions), new UTF8Encoding(false));
        File.Move(temp, path, true);
    }

    private static void DeleteActiveUnsafe()
    {
        var path = ActivePath();
        if (File.Exists(path)) File.Delete(path);
    }

    private static void ResetMemoryUnsafe()
    {
        CloseWriterUnsafe(); Anchors.Clear(); _partialPath = null; _runId = null;
        _sequence = 0; _previousHash = new string('0', 64); _attemptId = 0; _rollbackCount = 0;
        _lastDecisionSequence = null; _pendingResume = false; _pendingFromDecision = null; _pendingActionOutcome = null;
        _warningCount = 0;
    }

    private static void DisableUnsafe(string message, Exception? error)
    {
        _recordingDisabled = true;
        _lastError = message + (error is null ? "" : $": {error.GetType().Name}: {error.Message}");
        MainFile.Logger.Error("HumanRecorder disabled for this session: " + _lastError);
        try { WriteHealthUnsafe("disabled"); } catch { }
    }

    private static void WriteHealthUnsafe(string status)
    {
        WriteHealthValue(CreateHealthValueUnsafe(status));
    }

    private static void QueueHealthUpdateUnsafe(string status)
    {
        var value = CreateHealthValueUnsafe(status);
        ThreadPool.QueueUserWorkItem(_ =>
        {
            try { WriteHealthValue(value); } catch { }
        });
    }

    private static SortedDictionary<string, object?> CreateHealthValueUnsafe(string status)
    {
        return new SortedDictionary<string, object?>(StringComparer.Ordinal)
        {
            ["generation"] = Interlocked.Increment(ref _healthGeneration),
            ["schema_version"] = "human-recorder-health-0.1.0",
            ["recorder_version"] = MainFile.RecorderVersion,
            ["status"] = status,
            ["updated_at"] = DateTime.UtcNow.ToString("O"),
            ["run_id"] = _runId,
            ["partial_path"] = _partialPath,
            ["next_sequence"] = _sequence,
            ["last_decision_sequence"] = _lastDecisionSequence,
            ["attempt_id"] = _attemptId,
            ["last_error"] = _lastError,
            ["writer_metrics"] = _writer?.Metrics(),
            ["storage_root"] = Directory.GetParent(ResolveInbox())?.FullName ?? ResolveInbox(),
            ["storage_mode"] = _storageMode,
            ["storage_fallback_reason"] = _storageFallbackReason,
            ["hook_manifest"] = PatchRegistry.HookManifest()
        };
    }

    private static void WriteHealthValue(SortedDictionary<string, object?> value)
    {
        var generation = Convert.ToInt64(value["generation"]);
        var root = Directory.GetParent(ResolveInbox())?.FullName ?? ResolveInbox();
        Directory.CreateDirectory(root);
        var path = Path.Combine(root, "health.json");
        var temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        lock (HealthGate)
        {
            if (generation <= _lastHealthWritten) return;
            File.WriteAllText(temp, JsonSerializer.Serialize(value, JsonOptions), new UTF8Encoding(false));
            File.Move(temp, path, true);
            _lastHealthWritten = generation;
        }
    }

    private static void ArmPendingActionOutcomeUnsafe(
        long decisionSequence,
        string phase,
        Dictionary<string, object?> observation,
        Dictionary<string, object?> action)
    {
        var actionArgs = action.GetValueOrDefault("args") as Dictionary<string, object?>;
        var run = observation.GetValueOrDefault("run") as Dictionary<string, object?>;
        var ev = observation.GetValueOrDefault("event") as Dictionary<string, object?>;
        _pendingActionOutcome = new PendingActionOutcome(
            decisionSequence,
            _attemptId,
            phase,
            actionArgs?.GetValueOrDefault("potion_id")?.ToString(),
            actionArgs?.GetValueOrDefault("potion_instance_id")?.ToString(),
            ev?.GetValueOrDefault("id")?.ToString(),
            run?.GetValueOrDefault("room_type")?.ToString(),
            run?.GetValueOrDefault("total_floor"),
            run?.GetValueOrDefault("map_coord"));
    }

    private static void ResolvePendingActionOutcomeUnsafe(Dictionary<string, object?> observation)
    {
        var pending = _pendingActionOutcome;
        _pendingActionOutcome = null;
        if (pending is null || pending.AttemptId != _attemptId) return;

        var combat = observation.GetValueOrDefault("combat") as Dictionary<string, object?>;
        var enemies = combat?.GetValueOrDefault("enemies") as List<Dictionary<string, object?>>;
        if (enemies is null || enemies.Count == 0) return;

        var run = observation.GetValueOrDefault("run") as Dictionary<string, object?>;
        WriteUnsafe("engine_event", new Dictionary<string, object?>
        {
            ["run_id"] = _runId,
            ["attempt_id"] = _attemptId,
            ["after_decision_sequence"] = pending.SourceDecisionSequence,
            ["event_type"] = "potion_triggered_encounter",
            ["details"] = new Dictionary<string, object?>
            {
                ["source_decision_sequence"] = pending.SourceDecisionSequence,
                ["source_action_id"] = "use_potion",
                ["source_id"] = pending.SourceId,
                ["source_instance_id"] = pending.SourceInstanceId,
                ["origin_phase"] = pending.OriginPhase,
                ["origin_event_id"] = pending.OriginEventId,
                ["origin_room_type"] = pending.OriginRoomType,
                ["origin_floor"] = pending.OriginFloor,
                ["origin_map_coord"] = pending.OriginMapCoord,
                ["outcome_type"] = "encounter_started",
                ["encounter_ids"] = enemies.Select(x => x.GetValueOrDefault("id")).Where(x => x is not null).Distinct().ToList(),
                ["encounter_combat_ids"] = enemies.Select(x => x.GetValueOrDefault("combat_id")).Where(x => x is not null).ToList(),
                ["destination_room_type"] = run?.GetValueOrDefault("room_type"),
                ["destination_floor"] = run?.GetValueOrDefault("total_floor"),
                ["destination_map_coord"] = run?.GetValueOrDefault("map_coord")
            }
        });
    }

    private static void ReconcileAction(Dictionary<string, object?> observation, Dictionary<string, object?> action)
    {
        if (action["args"] is not Dictionary<string, object?> args) return;
        var actionId = action["action_id"]?.ToString();
        if (actionId == "play_card" && observation.GetValueOrDefault("combat") is Dictionary<string, object?> combat)
        {
            var objectId = args.GetValueOrDefault("card_object_id")?.ToString();
            var card = (combat.GetValueOrDefault("hand") as List<Dictionary<string, object?>>)?.FirstOrDefault(x =>
                x.GetValueOrDefault("instance_id")?.ToString()?.EndsWith(":" + objectId, StringComparison.Ordinal) == true);
            if (card is not null) args["card_instance_id"] = card["instance_id"];
            if (card is not null) args["card_lineage_id"] = card.GetValueOrDefault("lineage_id");
            var combatId = args.GetValueOrDefault("target_combat_id")?.ToString();
            var enemy = (combat.GetValueOrDefault("enemies") as List<Dictionary<string, object?>>)?.FirstOrDefault(x =>
                x.GetValueOrDefault("combat_id")?.ToString() == combatId);
            if (enemy is not null) args["target_index"] = enemy["index"];
        }
        if (actionId is "use_potion" or "discard_potion")
        {
            var objectId = args.GetValueOrDefault("potion_object_id")?.ToString();
            var potions = (observation.GetValueOrDefault("player") as Dictionary<string, object?>)?.GetValueOrDefault("potions")
                as List<Dictionary<string, object?>>;
            var potion = potions?.FirstOrDefault(x => !string.IsNullOrWhiteSpace(objectId)
                    && x.GetValueOrDefault("instance_id")?.ToString()?.EndsWith(":" + objectId, StringComparison.Ordinal) == true)
                ?? potions?.FirstOrDefault(x => x.GetValueOrDefault("id")?.ToString() == args.GetValueOrDefault("potion_id")?.ToString());
            if (potion is not null) args["potion_instance_id"] = potion["instance_id"];
            if (actionId == "use_potion")
            {
                args["usage_context"] = observation.GetValueOrDefault("foreground_phase")?.ToString()
                    ?? observation.GetValueOrDefault("phase")?.ToString();
                if (observation.GetValueOrDefault("run") is Dictionary<string, object?> originRun)
                {
                    args["origin_floor"] = originRun.GetValueOrDefault("total_floor");
                    args["origin_room_type"] = originRun.GetValueOrDefault("room_type");
                    args["origin_room_model_id"] = originRun.GetValueOrDefault("room_model_id");
                }
                if (observation.GetValueOrDefault("event") is Dictionary<string, object?> originEvent)
                    args["origin_event_id"] = originEvent.GetValueOrDefault("id");
            }
            if (actionId == "use_potion" && observation.GetValueOrDefault("combat") is Dictionary<string, object?> potionCombat)
            {
                var combatId = args.GetValueOrDefault("target_combat_id")?.ToString();
                var enemies = potionCombat.GetValueOrDefault("enemies") as List<Dictionary<string, object?>>;
                var enemy = enemies?.FirstOrDefault(x => !string.IsNullOrWhiteSpace(combatId)
                        && x.GetValueOrDefault("combat_id")?.ToString() == combatId)
                    ?? enemies?.FirstOrDefault(x => x.GetValueOrDefault("id")?.ToString() == args.GetValueOrDefault("target_id")?.ToString());
                if (enemy is not null) args["target_index"] = enemy["index"];
            }
        }
        if (actionId is "choose_card_reward" or "choose_card")
        {
            var section = observation.GetValueOrDefault(actionId == "choose_card_reward" ? "card_reward" : "card_select") as Dictionary<string, object?>;
            var cardId = args.GetValueOrDefault("card_id")?.ToString();
            var objectId = args.GetValueOrDefault("card_object_id")?.ToString();
            var card = (section?.GetValueOrDefault("cards") as List<Dictionary<string, object?>>)?.FirstOrDefault(x =>
                    !string.IsNullOrWhiteSpace(objectId)
                    && x.GetValueOrDefault("engine_object_ref")?.ToString() == objectId)
                ?? (section?.GetValueOrDefault("cards") as List<Dictionary<string, object?>>)?.FirstOrDefault(x =>
                    x.GetValueOrDefault("id")?.ToString() == cardId);
            if (card is not null)
            {
                args["card_instance_id"] = card["instance_id"];
                args["card_lineage_id"] = card.GetValueOrDefault("lineage_id");
            }
        }
        if (actionId is "confirm_card_selection" or "skip_card_selection")
        {
            var section = observation.GetValueOrDefault("card_select") as Dictionary<string, object?>;
            var selected = section?.GetValueOrDefault("selected_cards") as List<Dictionary<string, object?>>;
            if (selected is not null)
            {
                args["selected_card_instance_ids"] = selected.Select(x => x.GetValueOrDefault("instance_id")).ToList();
                args["selected_card_lineage_ids"] = selected.Select(x => x.GetValueOrDefault("lineage_id")).ToList();
            }
            args["selection_outcome"] = actionId == "skip_card_selection" ? "cancelled" : "confirmed";
            args["min_select"] = section?.GetValueOrDefault("min_select");
            args["max_select"] = section?.GetValueOrDefault("max_select");
        }
        if (actionId == "choose_relic")
        {
            var section = observation.GetValueOrDefault("relic_select") as Dictionary<string, object?>;
            var relics = section?.GetValueOrDefault("relics") as List<Dictionary<string, object?>>;
            var objectId = args.GetValueOrDefault("relic_object_id")?.ToString();
            var relic = relics?.FirstOrDefault(x => !string.IsNullOrWhiteSpace(objectId)
                    && x.GetValueOrDefault("instance_id")?.ToString()?.EndsWith(":" + objectId, StringComparison.Ordinal) == true)
                ?? relics?.FirstOrDefault(x => x.GetValueOrDefault("id")?.ToString() == args.GetValueOrDefault("relic_id")?.ToString());
            if (relic is not null)
            {
                args["relic_index"] = relic["index"];
                args["relic_instance_id"] = relic["instance_id"];
            }
        }
        if (actionId == "buy_shop_item")
        {
            var entries = observation.GetValueOrDefault("shop") as List<Dictionary<string, object?>>;
            var entry = entries?.FirstOrDefault(x =>
                x.GetValueOrDefault("id")?.ToString() == args.GetValueOrDefault("id")?.ToString()
                && ReflectionUtil.Int(x.GetValueOrDefault("cost")) == ReflectionUtil.Int(args.GetValueOrDefault("cost")));
            if (entry is not null) args["index"] = entry["index"];
        }
    }

    private static string ResolveInbox()
    {
        var configured = Environment.GetEnvironmentVariable("STS2_HUMAN_RECORDER_DIR");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            _storageMode = "environment_override";
            _storageFallbackReason = null;
            return Path.GetFullPath(configured);
        }
        return DefaultInboxPath.Value;
    }

    private static string ResolveDefaultInboxCore()
    {
        try
        {
            var assemblyPath = typeof(RecorderSession).Assembly.Location;
            var modRoot = Path.GetDirectoryName(assemblyPath);
            if (string.IsNullOrWhiteSpace(modRoot))
                throw new InvalidOperationException("Recorder assembly location is unavailable");
            var dataRoot = Path.Combine(modRoot, "记录数据");
            Directory.CreateDirectory(dataRoot);
            var probe = Path.Combine(dataRoot, ".write-test-" + Guid.NewGuid().ToString("N") + ".tmp");
            File.WriteAllText(probe, "ok", new UTF8Encoding(false));
            File.Delete(probe);
            _storageMode = "portable_mod_directory";
            return Path.Combine(dataRoot, "已完成记录");
        }
        catch (Exception ex)
        {
            _storageMode = "local_app_data_fallback";
            _storageFallbackReason = ex.GetType().Name + ": " + ex.Message;
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SlayTheSpire2", "HumanRecorder", "inbox");
        }
    }

    private static string ActivePath()
    {
        var root = Directory.GetParent(ResolveInbox())?.FullName ?? ResolveInbox();
        Directory.CreateDirectory(root);
        return Path.Combine(root, "active_run.json");
    }

    private static string LoadActorId()
    {
        var root = Directory.GetParent(ResolveInbox())?.FullName ?? ResolveInbox();
        Directory.CreateDirectory(root);
        var path = Path.Combine(root, "actor_id.txt");
        if (File.Exists(path)) return File.ReadAllText(path).Trim();
        var id = "anon-" + Guid.NewGuid().ToString("N");
        File.WriteAllText(path, id, new UTF8Encoding(false));
        return id;
    }

    private static object GameFingerprint()
    {
        var assembly = typeof(MegaCrit.Sts2.Core.Modding.ModInitializerAttribute).Assembly;
        var path = assembly.Location;
        var version = _gameAssemblyVersion ?? assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? assembly.GetName().Version?.ToString();
        var sha = _gameAssemblySha256;
        if (sha is null)
        {
            try { sha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant(); } catch { }
        }
        return new { version, assembly_sha256 = sha, assembly_file = Path.GetFileName(path), expected_game_version = "0.107.1", expected_build = "23811903" };
    }
}

internal readonly record struct RecorderUiState(string Status, int WarningCount);
