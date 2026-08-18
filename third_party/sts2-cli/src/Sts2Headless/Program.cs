using System.Reflection;
using System.Runtime.Loader;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sts2Headless;

class Program
{
    private sealed record CachedSaveEntry(
        string Json,
        RunSimulator.PreparedSave Prepared);

    private static readonly Dictionary<string, CachedSaveEntry> CachedSaves = new(StringComparer.Ordinal);

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    /// <summary>
    /// Locate the directory containing sts2.dll: STS2_LIB env, walk up from BaseDirectory, then BaseDirectory/lib.
    /// </summary>
    private static string ResolveLibDirectory()
    {
        var envLib = Environment.GetEnvironmentVariable("STS2_LIB");
        if (!string.IsNullOrWhiteSpace(envLib))
        {
            var p = Path.GetFullPath(envLib.Trim());
            if (Directory.Exists(p) && File.Exists(Path.Combine(p, "sts2.dll")))
                return p;
        }

        var dir = AppContext.BaseDirectory;
        for (var depth = 0; depth < 16 && !string.IsNullOrEmpty(dir); depth++)
        {
            var candidate = Path.Combine(dir, "lib");
            if (Directory.Exists(candidate) && File.Exists(Path.Combine(candidate, "sts2.dll")))
                return Path.GetFullPath(candidate);
            dir = Directory.GetParent(dir)?.FullName ?? "";
        }

        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "lib"));
    }

    static void Main(string[] args)
    {
        // Prevent unhandled exceptions from crashing the process
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            Console.Error.WriteLine($"[FATAL] Unhandled: {e.ExceptionObject}");
        };
        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            Console.Error.WriteLine($"[WARN] Unobserved task exception: {e.Exception?.Message}");
            e.SetObserved();
        };

        var libDir = ResolveLibDirectory();

        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            var path = Path.Combine(libDir, name.Name + ".dll");
            if (File.Exists(path))
                return ctx.LoadFromAssemblyPath(Path.GetFullPath(path));

            // Also check game directory (via STS2_GAME_DIR env var)
            var gameDir = Environment.GetEnvironmentVariable("STS2_GAME_DIR") ?? "";
            if (!string.IsNullOrEmpty(gameDir))
            {
                path = Path.Combine(gameDir, name.Name + ".dll");
                if (File.Exists(path))
                    return ctx.LoadFromAssemblyPath(path);
            }

            return null;
        };

        var sim = new RunSimulator();
        WriteLine(new Dictionary<string, object?> { ["type"] = "ready", ["version"] = "0.3.2" });

        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line)) continue;

            Dictionary<string, object?>? result;
            try
            {
                var cmd = JsonSerializer.Deserialize<JsonElement>(line);
                result = HandleCommand(sim, cmd);
            }
            catch (JsonException ex)
            {
                result = new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Invalid JSON: {ex.Message}" };
            }
            catch (Exception ex)
            {
                result = new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"{ex.GetType().Name}: {ex.Message}" };
            }

            if (result != null)
            {
                WriteLine(result);
                if (result.TryGetValue("type", out var resultTypeObj) &&
                    string.Equals(resultTypeObj as string, "quit_result", StringComparison.Ordinal))
                {
                    break;
                }
            }
        }
    }

    static Dictionary<string, object?>? HandleCommand(RunSimulator sim, JsonElement cmd)
    {
        var cmdType = cmd.GetProperty("cmd").GetString() ?? "";
        switch (cmdType)
        {
            case "start_run":
                List<string>? badges = null;
                if (cmd.TryGetProperty("badges", out var badgeArray))
                {
                    badges = new List<string>();
                    foreach (var badge in badgeArray.EnumerateArray())
                    {
                        var value = badge.GetString();
                        if (!string.IsNullOrWhiteSpace(value)) badges.Add(value);
                    }
                }
                return sim.StartRun(
                    cmd.TryGetProperty("character", out var ch) ? ch.GetString() ?? "Ironclad" : "Ironclad",
                    cmd.TryGetProperty("ascension", out var asc) ? asc.GetInt32() : 0,
                    cmd.TryGetProperty("seed", out var s) ? s.GetString() : null,
                    cmd.TryGetProperty("lang", out var lang) ? lang.GetString() ?? "en" : "en",
                    badges
                );

            case "action":
            {
                var action = cmd.GetProperty("action").GetString() ?? "";
                var actionArgs = cmd.TryGetProperty("args", out var argsElem)
                    ? ParseActionArgs(argsElem)
                    : null;
                return sim.ExecuteAction(action, actionArgs);
            }

            case "cache_save":
            {
                var cacheKey = cmd.TryGetProperty("name", out var nameElem)
                    ? nameElem.GetString()
                    : null;
                if (string.IsNullOrWhiteSpace(cacheKey))
                    return Error("cache_save requires a non-empty 'name'");

                var savePath = cmd.TryGetProperty("path", out var pathElem) ? pathElem.GetString() : null;
                var saveJson = cmd.TryGetProperty("json", out var jsonElem) ? jsonElem.GetString() : null;
                if (saveJson == null && savePath != null)
                {
                    if (!File.Exists(savePath))
                        return Error($"Save file not found: {savePath}");
                    saveJson = File.ReadAllText(savePath);
                }
                if (saveJson == null)
                    return Error("Provide 'path' or 'json' for cache_save");

                if (!sim.TryPrepareSave(saveJson, out var prepared, out var prepareError)
                    || prepared == null)
                {
                    return Error(prepareError);
                }
                CachedSaves[cacheKey] = new CachedSaveEntry(saveJson, prepared);
                return new Dictionary<string, object?>
                {
                    ["type"] = "ok",
                    ["name"] = cacheKey,
                    ["bytes"] = System.Text.Encoding.UTF8.GetByteCount(saveJson),
                    ["prepared"] = true,
                    ["prepared_packet_bytes"] = prepared.Packet.Length,
                };
            }

            case "restore_combat":
            {
                var includeProfile = cmd.TryGetProperty("profile", out var profileElem)
                    && profileElem.ValueKind == JsonValueKind.True;
                var compactPrefix = cmd.TryGetProperty("prefix_projection", out var projectionElem)
                    && string.Equals(
                        projectionElem.GetString(),
                        "compact",
                        StringComparison.OrdinalIgnoreCase);
                var profileWatch = Stopwatch.StartNew();
                var profileLastMs = 0.0;
                double MarkProfileStage()
                {
                    var elapsedMs = profileWatch.Elapsed.TotalMilliseconds;
                    var stageMs = elapsedMs - profileLastMs;
                    profileLastMs = elapsedMs;
                    return stageMs;
                }

                var cacheKey = cmd.TryGetProperty("cache", out var cacheElem)
                    ? cacheElem.GetString()
                    : null;
                if (string.IsNullOrWhiteSpace(cacheKey) || !CachedSaves.TryGetValue(cacheKey, out var cachedSave))
                    return Error($"Unknown cached save: {cacheKey}");
                var reusePreparedSave = cmd.TryGetProperty(
                        "reuse_prepared_save",
                        out var reusePreparedElem)
                    && reusePreparedElem.ValueKind == JsonValueKind.True;

                var loadLang = cmd.TryGetProperty("lang", out var langElem)
                    ? langElem.GetString() ?? "en"
                    : "en";
                var roomType = cmd.TryGetProperty("type", out var roomElem)
                    ? roomElem.GetString() ?? "combat"
                    : "combat";
                var encounter = cmd.TryGetProperty("encounter", out var encounterElem)
                    ? encounterElem.GetString()
                    : null;
                var eventId = cmd.TryGetProperty("event", out var eventElem)
                    ? eventElem.GetString()
                    : null;

                sim.CleanUp();
                var cleanupMs = MarkProfileStage();
                var state = reusePreparedSave
                    ? sim.LoadPreparedSave(cachedSave.Prepared, loadLang)
                    : sim.LoadSave(cachedSave.Json, loadLang);
                var loadSaveMs = MarkProfileStage();
                if (IsError(state)) return state;
                if (cmd.TryGetProperty("entry", out var entryElem))
                {
                    var entryType = entryElem.GetProperty("cmd").GetString() ?? "";
                    if (entryType == "action")
                    {
                        var action = entryElem.GetProperty("action").GetString() ?? "";
                        var actionArgs = entryElem.TryGetProperty("args", out var entryArgs)
                            ? ParseActionArgs(entryArgs)
                            : null;
                        state = sim.ExecuteAction(action, actionArgs, compactPrefix);
                    }
                    else if (entryType == "enter_room")
                    {
                        var entryRoomType = entryElem.TryGetProperty("type", out var entryRoom)
                            ? entryRoom.GetString() ?? roomType
                            : roomType;
                        var entryEncounter = entryElem.TryGetProperty("encounter", out var entryEncounterElem)
                            ? entryEncounterElem.GetString()
                            : encounter;
                        var entryEvent = entryElem.TryGetProperty("event", out var entryEventElem)
                            ? entryEventElem.GetString()
                            : eventId;
                        state = sim.EnterRoom(
                            entryRoomType,
                            entryEncounter,
                            entryEvent,
                            compactPrefix);
                    }
                    else
                    {
                        return Error($"Unsupported restore entry command: {entryType}");
                    }
                }
                else
                {
                    state = sim.EnterRoom(roomType, encounter, eventId, compactPrefix);
                }
                var enterCombatMs = MarkProfileStage();
                if (IsError(state)) return state;

                if (cmd.TryGetProperty("prefix", out var prefixElem))
                {
                    foreach (var step in prefixElem.EnumerateArray())
                    {
                        var action = step.GetProperty("action").GetString() ?? "";
                        var actionArgs = step.TryGetProperty("args", out var stepArgs)
                            ? ParseActionArgs(stepArgs)
                            : null;
                        state = sim.ExecuteAction(action, actionArgs, compactPrefix);
                        if (IsError(state)) return state;
                    }
                }
                var prefixReplayMs = MarkProfileStage();

                if (cmd.TryGetProperty("draw_order", out var drawOrderElem))
                {
                    var cards = new List<string>();
                    foreach (var card in drawOrderElem.EnumerateArray())
                        cards.Add(card.GetString() ?? "");
                    var drawResult = sim.SetDrawOrder(cards);
                    if (IsError(drawResult)) return drawResult;
                }
                var drawOrderMs = MarkProfileStage();
                if (cmd.TryGetProperty("suffix", out var suffixElem))
                {
                    foreach (var step in suffixElem.EnumerateArray())
                    {
                        var action = step.GetProperty("action").GetString() ?? "";
                        var actionArgs = step.TryGetProperty("args", out var stepArgs)
                            ? ParseActionArgs(stepArgs)
                            : null;
                        state = sim.ExecuteAction(action, actionArgs, compactPrefix);
                        if (IsError(state)) return state;
                    }
                }
                var suffixReplayMs = MarkProfileStage();
                var finalProjectionMs = 0.0;
                if (compactPrefix)
                {
                    // Intermediate responses intentionally omitted the full
                    // visible state.  Project it exactly once after the prefix
                    // and any deterministic draw-order override are complete.
                    state = sim.GetState();
                    if (IsError(state)) return state;
                    finalProjectionMs = MarkProfileStage();
                }
                if (includeProfile)
                {
                    state["_profile_ms"] = new Dictionary<string, object?>
                    {
                        ["cleanup"] = cleanupMs,
                        ["load_save"] = loadSaveMs,
                        ["enter_combat"] = enterCombatMs,
                        ["prefix_replay"] = prefixReplayMs,
                        ["set_draw_order"] = drawOrderMs,
                        ["suffix_replay"] = suffixReplayMs,
                        ["final_projection"] = finalProjectionMs,
                        ["server_pre_serialize_total"] = profileWatch.Elapsed.TotalMilliseconds,
                    };
                }
                return state;
            }

            case "load_save":
            {
                var savePath = cmd.TryGetProperty("path", out var sp) ? sp.GetString() : null;
                var saveJson = cmd.TryGetProperty("json", out var sj) ? sj.GetString() : null;
                if (saveJson == null && savePath != null)
                {
                    if (!File.Exists(savePath))
                        return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Save file not found: {savePath}" };
                    saveJson = File.ReadAllText(savePath);
                }
                if (saveJson == null)
                    return new Dictionary<string, object?> { ["type"] = "error", ["message"] = "Provide 'path' or 'json' for load_save" };
                var loadLang = cmd.TryGetProperty("lang", out var le) ? (le.GetString() ?? "en") : "en";
                return sim.LoadSave(saveJson, loadLang);
            }

            case "reload_save":
            {
                var savePath = cmd.TryGetProperty("path", out var sp) ? sp.GetString() : null;
                var saveJson = cmd.TryGetProperty("json", out var sj) ? sj.GetString() : null;
                if (saveJson == null && savePath != null)
                {
                    if (!File.Exists(savePath))
                        return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Save file not found: {savePath}" };
                    saveJson = File.ReadAllText(savePath);
                }
                if (saveJson == null)
                    return new Dictionary<string, object?> { ["type"] = "error", ["message"] = "Provide 'path' or 'json' for reload_save" };
                var loadLang = cmd.TryGetProperty("lang", out var le) ? (le.GetString() ?? "en") : "en";
                sim.CleanUp();
                return sim.LoadSave(saveJson, loadLang);
            }
            case "get_map":
                return sim.GetFullMap();

            case "get_state":
                return sim.GetState();

            case "set_player":
            {
                var args = new Dictionary<string, JsonElement>();
                foreach (var prop in cmd.EnumerateObject())
                    if (prop.Name != "cmd") args[prop.Name] = prop.Value;
                return sim.SetPlayer(args);
            }

            case "enter_room":
            {
                var roomType = cmd.TryGetProperty("type", out var rt) ? rt.GetString() ?? "" : "";
                var encounter = cmd.TryGetProperty("encounter", out var enc) ? enc.GetString() : null;
                var eventId = cmd.TryGetProperty("event", out var ev) ? ev.GetString() : null;
                return sim.EnterRoom(roomType, encounter, eventId);
            }

            case "set_draw_order":
            {
                var cards = new List<string>();
                if (cmd.TryGetProperty("cards", out var cardsArr))
                    foreach (var c in cardsArr.EnumerateArray())
                        cards.Add(c.GetString() ?? "");
                return sim.SetDrawOrder(cards);
            }

            case "write_continue_save":
            {
                var outputPath = cmd.TryGetProperty("path", out var op) ? op.GetString() : null;
                return sim.SaveCheckpoint(outputPath);
            }

            case "quit":
            {
                var outputPath = cmd.TryGetProperty("path", out var op) ? op.GetString() : null;
                if (!string.IsNullOrEmpty(outputPath))
                {
                    var saveResult = sim.SaveCheckpoint(outputPath);
                    bool saveOk = saveResult.TryGetValue("success", out var sObj) && sObj is bool b && b;
                    if (!saveOk)
                    {
                        // Save failed — do NOT clean up so the caller can retry with a different path.
                        return new Dictionary<string, object?>
                        {
                            ["type"] = "save_error",
                            ["save"] = saveResult,
                        };
                    }
                    sim.CleanUp();
                    return new Dictionary<string, object?>
                    {
                        ["type"] = "quit_result",
                        ["success"] = true,
                        ["save"] = saveResult,
                    };
                }
                sim.CleanUp();
                return new Dictionary<string, object?>
                {
                    ["type"] = "quit_result",
                    ["success"] = true,
                    ["save"] = null,
                };
            }

            default:
                return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Unknown command: {cmdType}" };
        }
    }

    private static Dictionary<string, object?> ParseActionArgs(JsonElement argsElem)
    {
        var actionArgs = new Dictionary<string, object?>();
        foreach (var prop in argsElem.EnumerateObject())
        {
            actionArgs[prop.Name] = prop.Value.ValueKind switch
            {
                JsonValueKind.Number => prop.Value.TryGetInt32(out var value)
                    ? value
                    : prop.Value.GetDouble(),
                JsonValueKind.String => prop.Value.GetString(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                _ => prop.Value.ToString(),
            };
        }
        return actionArgs;
    }

    private static bool IsError(Dictionary<string, object?> result) =>
        result.TryGetValue("type", out var type) && string.Equals(type as string, "error", StringComparison.Ordinal);

    private static Dictionary<string, object?> Error(string message) =>
        new() { ["type"] = "error", ["message"] = message };

    static void WriteLine(Dictionary<string, object?> data)
    {
        Console.Out.WriteLine(JsonSerializer.Serialize(data, JsonOpts));
        Console.Out.Flush();
    }
}
