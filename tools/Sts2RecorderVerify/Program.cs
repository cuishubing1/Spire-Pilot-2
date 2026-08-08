using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;

Environment.SetEnvironmentVariable("STS2_HUMAN_RECORDER_DISABLE_GODOT_LOOKUP", "1");

if (args.Length != 2)
{
    Console.Error.WriteLine("usage: Sts2RecorderVerify <game-data-dir> <HumanRecorder.dll>");
    return 2;
}
var dataDir = Path.GetFullPath(args[0]);
AssemblyLoadContext.Default.Resolving += (_, name) =>
{
    foreach (var root in new[] { dataDir, Path.GetDirectoryName(Path.GetFullPath(args[1]))! })
    {
        var candidate = Path.Combine(root, name.Name + ".dll");
        if (File.Exists(candidate)) return AssemblyLoadContext.Default.LoadFromAssemblyPath(candidate);
    }
    return null;
};
var game = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(dataDir, "sts2.dll"));
var mod = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.GetFullPath(args[1]));
if (Environment.GetEnvironmentVariable("STS2_RECORDER_ACTION_AUDIT") == "1")
{
    var requestedPattern = Environment.GetEnvironmentVariable("STS2_RECORDER_ACTION_FILTER");
    var typePattern = string.IsNullOrWhiteSpace(requestedPattern)
        ? new[] { "Bundle", "Reward", "Potion", "Treasure", "Terminal", "CardSelect", "Merchant", "RestSite", "Event" }
        : requestedPattern.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    var methodPattern = new[] { "Select", "Skip", "Discard", "Proceed", "Claim", "Choose", "Confirm", "Take", "Obtain", "Click", "Release", "Purchase", "Hide", "Play", "EndTurn", "Remove", "Relic", "Open", "Get", "Amount", "Damage", "Heal" };
    var candidates = game.GetTypes()
        .Where(type => type.FullName is not null && typePattern.Any(token => type.FullName.Contains(token, StringComparison.OrdinalIgnoreCase)))
        .Select(type => new
        {
            type = type.FullName,
            properties = type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
                .Select(property => new { name = property.Name, type = property.PropertyType.FullName }).OrderBy(property => property.name).ToArray(),
            fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
                .Select(field => new { name = field.Name, type = field.FieldType.FullName }).OrderBy(field => field.name).ToArray(),
            methods = type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
                .Where(method => methodPattern.Any(token => method.Name.Contains(token, StringComparison.OrdinalIgnoreCase)))
                .Select(method => new
                {
                    name = method.Name,
                    declaring_type = method.DeclaringType?.FullName,
                    is_static = method.IsStatic,
                    is_abstract = method.IsAbstract,
                    has_body = method.GetMethodBody() is not null,
                    return_type = method.ReturnType.FullName,
                    parameters = method.GetParameters().Select(parameter => new { name = parameter.Name, type = parameter.ParameterType.FullName }).ToArray()
                }).OrderBy(method => method.name).ToArray()
        })
        .Where(candidate => candidate.methods.Length > 0)
        .OrderBy(candidate => candidate.type).ToArray();
    Console.WriteLine(JsonSerializer.Serialize(candidates, new JsonSerializerOptions { WriteIndented = true }));
    return 0;
}
var registry = mod.GetType("Sts2HumanRecorder.PatchRegistry", true)!;
var specs = (Array)registry.GetField("Specs", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
var checkedHooks = new List<object>();
var failures = new List<string>();
foreach (var spec in specs)
{
    var typeName = (string)spec!.GetType().GetProperty("TypeName")!.GetValue(spec)!;
    var methodName = (string)spec.GetType().GetProperty("MethodName")!.GetValue(spec)!;
    var required = (bool)spec.GetType().GetProperty("Required")!.GetValue(spec)!;
    var type = game.GetType(typeName);
    var count = type is null ? 0 : methodName == ".ctor"
        ? type.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance).Length
        : type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
            .Count(x => x.Name == methodName && x.DeclaringType == type);
    if (required && count == 0) failures.Add(typeName + "." + methodName);
    checkedHooks.Add(new { type = typeName, method = methodName, overloads = count, required });
}
foreach (var (typeName, methodName) in new[]
{
    ("MegaCrit.Sts2.Core.Runs.RunManager", "SetUpNewSingleplayer"),
    ("MegaCrit.Sts2.Core.Runs.RunManager", "SetUpSavedSingleplayer"),
    ("MegaCrit.Sts2.Core.Runs.RunManager", "OnEnded"),
    ("MegaCrit.Sts2.Core.Runs.RunManager", "Abandon"),
    ("MegaCrit.Sts2.Core.Commands.CreatureCmd", "Heal"),
    ("MegaCrit.Sts2.Core.Models.RelicModel", "Flash")
})
{
    var type = game.GetType(typeName);
    var count = type?.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static).Count(x => x.Name == methodName) ?? 0;
    if (count == 0) failures.Add(typeName + "." + methodName);
    checkedHooks.Add(new { type = typeName, method = methodName, overloads = count, required = true });
}
string harmonySmoke;
try
{
    var harmonyAssembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(dataDir, "0Harmony.dll"));
    var harmonyType = harmonyAssembly.GetType("HarmonyLib.Harmony", true)!;
    var harmony = Activator.CreateInstance(harmonyType, "HumanRecorder.StaticSmoke")!;
    var install = registry.GetMethod("Install", BindingFlags.Static | BindingFlags.Public)
        ?? throw new MissingMethodException("PatchRegistry.Install");
    install.Invoke(null, new[] { harmony });
    harmonyType.GetMethod("UnpatchSelf", BindingFlags.Instance | BindingFlags.Public)?.Invoke(harmony, null);
    harmonySmoke = "PASS";
}
catch (Exception ex)
{
    harmonySmoke = "FAIL: " + (ex.InnerException ?? ex).Message;
    failures.Add("Harmony patch smoke: " + harmonySmoke);
}
string lifecycleSmoke;
var tempRoot = Path.Combine(Path.GetTempPath(), "sts2-human-recorder-smoke-" + Guid.NewGuid().ToString("N"));
var previousInbox = Environment.GetEnvironmentVariable("STS2_HUMAN_RECORDER_DIR");
try
{
    var inbox = Path.Combine(tempRoot, "inbox");
    Environment.SetEnvironmentVariable("STS2_HUMAN_RECORDER_DIR", inbox);
    var session = mod.GetType("Sts2HumanRecorder.RecorderSession", true)!;
    session.GetMethod("Initialize", BindingFlags.Static | BindingFlags.Public)!.Invoke(null, null);
    session.GetMethod("StartNewRun", BindingFlags.Static | BindingFlags.Public)!.Invoke(null, null);
    var getUiState = session.GetMethod("GetUiState", BindingFlags.Static | BindingFlags.NonPublic)!;
    string UiStatus() => getUiState.Invoke(null, null)!.GetType().GetProperty("Status")!
        .GetValue(getUiState.Invoke(null, null))!.ToString()!;
    if (UiStatus() != "recording") throw new InvalidDataException("overlay did not report active recording");
    var recordAction = session.GetMethod("RecordAction", BindingFlags.Static | BindingFlags.Public)!;
    var firstCommit = recordAction.Invoke(null, new object?[] { "map_select", "select_map_node", null, Array.Empty<object>() });
    firstCommit?.GetType().GetMethod("Complete")!.Invoke(firstCommit, new object[] { "method_returned" });
    var armPotionOutcome = session.GetMethod("ArmPendingActionOutcomeUnsafe", BindingFlags.Static | BindingFlags.NonPublic)!;
    var resolvePotionOutcome = session.GetMethod("ResolvePendingActionOutcomeUnsafe", BindingFlags.Static | BindingFlags.NonPublic)!;
    armPotionOutcome.Invoke(null, new object?[]
    {
        2L, "event_choice",
        new Dictionary<string, object?>
        {
            ["run"] = new Dictionary<string, object?> { ["room_type"] = "Event", ["total_floor"] = 39 },
            ["event"] = new Dictionary<string, object?> { ["id"] = "EVENT.FAKE_MERCHANT" }
        },
        new Dictionary<string, object?>
        {
            ["args"] = new Dictionary<string, object?>
            {
                ["potion_id"] = "POTION.FOUL_POTION", ["potion_instance_id"] = "potion:1:smoke"
            }
        }
    });
    resolvePotionOutcome.Invoke(null, new object?[]
    {
        new Dictionary<string, object?>
        {
            ["run"] = new Dictionary<string, object?> { ["room_type"] = "Event", ["total_floor"] = 39 },
            ["combat"] = new Dictionary<string, object?>
            {
                ["enemies"] = new List<Dictionary<string, object?>>
                {
                    new() { ["id"] = "MONSTER.FAKE_MERCHANT_MONSTER", ["combat_id"] = "enemy:0" }
                }
            }
        }
    });
    session.GetMethod("RecordEngineEvent", BindingFlags.Static | BindingFlags.Public)!.Invoke(null,
        new object?[] { "heal_requested", new Dictionary<string, object?> { ["requested_amount"] = 10m } });
    session.GetMethod("Suspend", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, new object?[] { "smoke_restart" });
    session.GetMethod("ResumeRun", BindingFlags.Static | BindingFlags.Public)!.Invoke(null, null);
    if (UiStatus() != "restoring") throw new InvalidDataException("overlay did not report SL restoration");
    var secondCommit = recordAction.Invoke(null, new object?[] { "map_select", "select_map_node", null, Array.Empty<object>() });
    secondCommit?.GetType().GetMethod("Complete")!.Invoke(secondCommit, new object[] { "method_returned" });
    session.GetMethod("EndRun", BindingFlags.Static | BindingFlags.Public)!.Invoke(null, new object?[] { "abandoned", false });
    if (UiStatus() != "idle") throw new InvalidDataException("overlay did not return to idle after run end");
    var files = Directory.GetFiles(inbox, "*.jsonl");
    if (files.Length != 1 || Directory.GetFiles(inbox, "*.partial").Length != 0)
        throw new InvalidDataException($"expected one sealed logical-run file, got {files.Length}");
    var rows = File.ReadLines(files[0]).Select(line => JsonDocument.Parse(line)).ToList();
    var types = rows.Select(x => x.RootElement.GetProperty("record_type").GetString()).ToList();
    if (types.Count(x => x == "rollback") != 1 || types.Count(x => x == "decision") != 2
        || types.Count(x => x == "engine_event") != 2)
        throw new InvalidDataException("resume/events were not represented in one logical run");
    var potionOutcome = rows.Single(x => x.RootElement.GetProperty("record_type").GetString() == "engine_event"
        && x.RootElement.GetProperty("payload").GetProperty("event_type").GetString() == "potion_triggered_encounter");
    var potionOutcomeDetails = potionOutcome.RootElement.GetProperty("payload").GetProperty("details");
    if (potionOutcomeDetails.GetProperty("source_id").GetString() != "POTION.FOUL_POTION"
        || potionOutcomeDetails.GetProperty("origin_event_id").GetString() != "EVENT.FAKE_MERCHANT"
        || potionOutcomeDetails.GetProperty("encounter_ids")[0].GetString() != "MONSTER.FAKE_MERCHANT_MONSTER")
        throw new InvalidDataException("potion-triggered encounter causal record is incomplete");
    var attempts = rows.Where(x => x.RootElement.GetProperty("record_type").GetString() == "decision")
        .Select(x => x.RootElement.GetProperty("payload").GetProperty("attempt_id").GetInt32()).ToArray();
    if (!attempts.SequenceEqual(new[] { 0, 1 })) throw new InvalidDataException("attempt ids are not continuous");
    foreach (var row in rows) row.Dispose();
    lifecycleSmoke = "PASS";
}
catch (Exception ex)
{
    lifecycleSmoke = "FAIL: " + (ex.InnerException ?? ex).Message;
    failures.Add("Recorder lifecycle smoke: " + lifecycleSmoke);
}
finally
{
    Environment.SetEnvironmentVariable("STS2_HUMAN_RECORDER_DIR", previousInbox);
    if (Directory.Exists(tempRoot)) Directory.Delete(tempRoot, true);
}
string matcherSmoke;
try
{
    var matcherType = mod.GetType("Sts2HumanRecorder.StateFingerprint", true)!;
    var anchorType = mod.GetType("Sts2HumanRecorder.DecisionAnchor", true)!;
    var build = matcherType.GetMethods(BindingFlags.Static | BindingFlags.Public)
        .Single(x => x.Name == "Build" && x.GetParameters()[0].ParameterType == typeof(Dictionary<string, object?>));
    var match = matcherType.GetMethod("Match", BindingFlags.Static | BindingFlags.Public)!;
    Dictionary<string, object?> Observation(int floor, int hp, string instanceId) => new()
    {
        ["phase"] = "combat_play",
        ["run"] = new Dictionary<string, object?>
        {
            ["act"] = 1, ["total_floor"] = floor, ["room_type"] = "Boss",
            ["map_coord"] = new Dictionary<string, object?> { ["col"] = 2, ["row"] = 15 }
        },
        ["player"] = new Dictionary<string, object?>
        {
            ["character_id"] = "CHARACTER.IRONCLAD", ["hp"] = hp, ["max_hp"] = 80, ["gold"] = 99,
            ["deck"] = new List<Dictionary<string, object?>>(), ["relics"] = new List<Dictionary<string, object?>>(),
            ["potions"] = new List<Dictionary<string, object?>>()
        },
        ["combat"] = new Dictionary<string, object?>
        {
            ["round"] = 1, ["turn"] = 1, ["turn_phase"] = "Play", ["energy"] = 3,
            ["hand"] = new List<Dictionary<string, object?>>
            {
                new() { ["id"] = "CARD.STRIKE_IRONCLAD", ["instance_id"] = instanceId, ["cost"] = 1 }
            },
            ["enemies"] = new List<Dictionary<string, object?>>()
        },
        ["legal_actions"] = new List<object>()
    };
    var anchor = build.Invoke(null, new object?[] { Observation(17, 30, "old"), 10L, 0 })!;
    var listType = typeof(List<>).MakeGenericType(anchorType);
    var history = (System.Collections.IList)Activator.CreateInstance(listType)!;
    history.Add(anchor);
    string Quality(object current) => match.Invoke(null, new[] { current, history, 0 })!
        .GetType().GetProperty("Quality")!.GetValue(match.Invoke(null, new[] { current, history, 0 }))!.ToString()!;
    var exact = build.Invoke(null, new object?[] { Observation(17, 30, "new"), 20L, 1 })!;
    var room = build.Invoke(null, new object?[] { Observation(17, 29, "new"), 21L, 1 })!;
    var missing = build.Invoke(null, new object?[] { Observation(18, 29, "new"), 22L, 1 })!;
    if (Quality(exact) != "exact" || Quality(room) != "room_entry" || Quality(missing) != "unmatched")
        throw new InvalidDataException("exact/room-entry/unmatched matcher degradation is incorrect");
    matcherSmoke = "PASS";
}
catch (Exception ex)
{
    matcherSmoke = "FAIL: " + (ex.InnerException ?? ex).Message;
    failures.Add("State matcher smoke: " + matcherSmoke);
}
string isolationSmoke;
try
{
    var session = mod.GetType("Sts2HumanRecorder.RecorderSession", true)!;
    var inferVictory = session.GetMethod("InferVictory", BindingFlags.Static | BindingFlags.NonPublic)!;
    Dictionary<string, object?> Terminal(int hp) => new()
    {
        ["player"] = new Dictionary<string, object?> { ["hp"] = hp },
        ["run"] = new Dictionary<string, object?> { ["is_game_over"] = false }
    };
    if (!Equals(inferVictory.Invoke(null, new object?[] { "game_ended", Terminal(42) }), true)
        || !Equals(inferVictory.Invoke(null, new object?[] { "game_ended", Terminal(0) }), false))
        throw new InvalidDataException("terminal positive-HP victory inference is incorrect");

    var provenance = mod.GetType("Sts2HumanRecorder.ContentProvenance", true)!;
    var classify = provenance.GetMethod("ClassifyDecision", BindingFlags.Static | BindingFlags.Public)!;
    Dictionary<string, object?> Scoped(string quality, string sourceKind) => new()
    {
        ["capture_quality"] = quality,
        ["player"] = new Dictionary<string, object?>
        {
            ["deck"] = new List<Dictionary<string, object?>> { new() { ["source_kind"] = sourceKind } }
        }
    };
    var emptyAction = new Dictionary<string, object?>();
    if ((string)classify.Invoke(null, new object?[] { Scoped("complete", "mod"), emptyAction })! != "modded"
        || (string)classify.Invoke(null, new object?[] { Scoped("complete", "base_game"), emptyAction })! != "base_game"
        || (string)classify.Invoke(null, new object?[] { Scoped("partial", "base_game"), emptyAction })! != "unknown")
        throw new InvalidDataException("Mod/base/unknown content isolation is incorrect");
    isolationSmoke = "PASS";
}
catch (Exception ex)
{
    isolationSmoke = "FAIL: " + (ex.InnerException ?? ex).Message;
    failures.Add("Victory/Mod isolation smoke: " + isolationSmoke);
}
string combatSchemaSmoke;
var intentApi = new List<object>();
try
{
    var attackIntent = game.GetType("MegaCrit.Sts2.Core.MonsterMoves.Intents.AttackIntent", true)!;
    foreach (var methodName in new[] { "GetSingleDamage", "GetTotalDamage" })
    {
        var methods = attackIntent.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Where(x => x.Name == methodName).ToArray();
        if (methods.Length == 0) throw new MissingMethodException(attackIntent.FullName, methodName);
        intentApi.Add(new
        {
            type = attackIntent.FullName, method = methodName, overloads = methods.Length,
            parameter_counts = methods.Select(x => x.GetParameters().Length).Distinct().Order().ToArray()
        });
    }
    var repeats = attackIntent.GetProperty("Repeats", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
    if (repeats is null) throw new MissingMemberException(attackIntent.FullName, "Repeats");
    var deathBlowIntent = game.GetType("MegaCrit.Sts2.Core.MonsterMoves.Intents.DeathBlowIntent", true)!;
    if (!attackIntent.IsAssignableFrom(deathBlowIntent))
        throw new InvalidDataException("DeathBlowIntent is no longer an AttackIntent");
    var abstractIntent = game.GetType("MegaCrit.Sts2.Core.MonsterMoves.Intents.AbstractIntent", true)!;
    foreach (var methodName in new[] { "GetIntentLabel", "GetIntentDescription" })
        if (!abstractIntent.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Any(x => x.Name == methodName && x.GetParameters().Length == 2))
            throw new MissingMethodException(abstractIntent.FullName, methodName);
    var locString = game.GetType("MegaCrit.Sts2.Core.Localization.LocString", true)!;
    if (locString.GetProperty("Variables", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance) is null)
        throw new MissingMemberException(locString.FullName, "Variables");

    var exporter = mod.GetType("Sts2HumanRecorder.StateExporter", true)!;
    var complete = exporter.GetMethod("IntentDamageComplete", BindingFlags.Static | BindingFlags.NonPublic)!;
    Dictionary<string, object?> Enemy(bool intendsAttack, params Dictionary<string, object?>[] intents) => new()
    {
        ["intends_attack"] = intendsAttack,
        ["intent"] = intents.ToList()
    };
    bool IsComplete(Dictionary<string, object?> enemy) => (bool)complete.Invoke(null, new object?[] { enemy })!;
    if (!IsComplete(Enemy(false))
        || !IsComplete(Enemy(true, new Dictionary<string, object?> { ["type"] = "Attack", ["is_attack"] = true, ["damage"] = 12, ["hits"] = 2 }))
        || !IsComplete(Enemy(true, new Dictionary<string, object?> { ["type"] = "DeathBlow", ["is_attack"] = true, ["damage"] = 999, ["hits"] = 1 }))
        || IsComplete(Enemy(true, new Dictionary<string, object?> { ["type"] = "Attack", ["hits"] = 2 }))
        || IsComplete(Enemy(true, new Dictionary<string, object?> { ["type"] = "DeathBlow", ["is_attack"] = true }))
        || IsComplete(Enemy(true, new Dictionary<string, object?> { ["type"] = "Buff" })))
        throw new InvalidDataException("attack intent damage completeness does not fail closed");

    var card = game.GetType("MegaCrit.Sts2.Core.Models.CardModel", true)!;
    var preview = card.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
        .Where(x => x.Name == "UpdateDynamicVarPreview").ToArray();
    if (preview.Length == 0) throw new MissingMethodException(card.FullName, "UpdateDynamicVarPreview");
    var previewMode = game.GetType("MegaCrit.Sts2.Core.Entities.Cards.CardPreviewMode", true)!;
    foreach (var requiredName in new[] { "Normal", "MultiCreatureTargeting" })
        if (!Enum.GetNames(previewMode).Contains(requiredName))
            throw new MissingMemberException(previewMode.FullName, requiredName);

    combatSchemaSmoke = "PASS";
}
catch (Exception ex)
{
    combatSchemaSmoke = "FAIL: " + (ex.InnerException ?? ex).Message;
    failures.Add("Combat schema/API smoke: " + combatSchemaSmoke);
}
string nativeModelStateSmoke;
try
{
    var savedProperties = game.GetType("MegaCrit.Sts2.Core.Saves.Runs.SavedProperties", true)!;
    foreach (var field in new[] { "ints", "bools", "strings", "intArrays", "modelIds", "cards", "cardArrays" })
        if (savedProperties.GetField(field, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance) is null)
            throw new MissingFieldException(savedProperties.FullName, field);
    var cardModel = game.GetType("MegaCrit.Sts2.Core.Models.CardModel", true)!;
    foreach (var property in new[] { "Enchantment", "Affliction", "EnergyCost", "CurrentUpgradeLevel" })
        if (cardModel.GetProperty(property, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance) is null)
            throw new MissingMemberException(cardModel.FullName, property);
    var relicModel = game.GetType("MegaCrit.Sts2.Core.Models.RelicModel", true)!;
    foreach (var property in new[] { "StackCount", "Status", "ShowCounter", "DisplayAmount", "IsUsedUp" })
        if (relicModel.GetProperty(property, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance) is null)
            throw new MissingMemberException(relicModel.FullName, property);

    var native = mod.GetType("Sts2HumanRecorder.NativeModelState", true)!;
    var encodeSaved = native.GetMethod("EncodeSavedProperties", BindingFlags.Public | BindingFlags.Static)!;
    var fakeProps = new RecorderFakeSavedProperties();
    fakeProps.ints.Add(new RecorderFakeSavedProperty<int>("TimesUsed", 2));
    fakeProps.bools.Add(new RecorderFakeSavedProperty<bool>("UsedThisCombat", true));
    fakeProps.intArrays.Add(new RecorderFakeSavedProperty<int[]>("Coords", new[] { 1, 3 }));
    var encoded = (List<Dictionary<string, object?>>)encodeSaved.Invoke(null,
        new object?[] { fakeProps, "SilverCrucible", "public" })!;
    if (encoded.Count != 3
        || encoded.Single(row => row["key"]?.ToString() == "TimesUsed")["int_value"]?.ToString() != "2"
        || encoded.Single(row => row["key"]?.ToString() == "UsedThisCombat")["lifecycle"]?.ToString() != "combat"
        || encoded.Single(row => row["key"]?.ToString() == "Coords")["value_type"]?.ToString() != "int_list")
        throw new InvalidDataException("native SavedProperties encoding failed");

    var exporter = mod.GetType("Sts2HumanRecorder.StateExporter", true)!;
    var cardList = exporter.GetMethod("CardList", BindingFlags.Static | BindingFlags.NonPublic)!;
    var fakeCard = new RecorderFakeCard("CARD.NATIVE", new RecorderFakeEnchantment("ENCHANTMENT.GLAM", 1));
    var cards = (List<Dictionary<string, object?>>)cardList.Invoke(null,
        new object?[] { new[] { fakeCard }, "verify", false, null, 0 })!;
    var cardRow = cards.Single();
    var nestedEnchantment = cardRow["enchantment"] as Dictionary<string, object?>;
    if (nestedEnchantment?["id"]?.ToString() != "ENCHANTMENT.GLAM"
        || cardRow.ContainsKey("enchantment_id") || cardRow.ContainsKey("affliction_id")
        || cardRow["energy_cost"] is not Dictionary<string, object?>
        || cardRow["runtime_flags"] is not Dictionary<string, object?>)
        throw new InvalidDataException("nested card state schema failed");

    var stateRegistry = mod.GetType("Sts2HumanRecorder.ContentStateRegistry", true)!;
    if (stateRegistry.GetField("Version", BindingFlags.Public | BindingFlags.Static)?.GetRawConstantValue()?.ToString()
        != "sts2-v0.107.1-state-projection-1")
        throw new InvalidDataException("state projection registry version missing");
    nativeModelStateSmoke = "PASS";
}
catch (Exception ex)
{
    nativeModelStateSmoke = "FAIL: " + (ex.InnerException ?? ex).Message;
    failures.Add("Native model-state smoke: " + nativeModelStateSmoke);
}
string actionCoverageSmoke;
var semanticActions = new List<object>();
try
{
    var actual = specs.Cast<object>().Select(spec => (
        Phase: (string)spec.GetType().GetProperty("Phase")!.GetValue(spec)!,
        Action: (string)spec.GetType().GetProperty("ActionId")!.GetValue(spec)!,
        Type: (string)spec.GetType().GetProperty("TypeName")!.GetValue(spec)!,
        Method: (string)spec.GetType().GetProperty("MethodName")!.GetValue(spec)!
    )).ToList();
    var expected = new (string Phase, string Action)[]
    {
        ("map_select", "select_map_node"), ("combat_play", "play_card"), ("combat_play", "end_turn"),
        ("combat_play", "use_potion"), ("potion_manage", "discard_potion"),
        ("event_choice", "choose_event_option"), ("rest_site", "choose_rest_option"),
        ("shop", "buy_shop_item"), ("shop", "remove_card"), ("shop", "leave_shop"),
        ("card_reward", "choose_card_reward"), ("card_reward", "choose_reward_alternative"),
        ("card_reward", "skip"), ("card_select", "choose_card"),
        ("card_select", "confirm_card_selection"), ("card_select", "skip_card_selection"),
        ("bundle_select", "select_bundle"), ("relic_select", "choose_relic"),
        ("relic_select", "skip"), ("reward_select", "select_reward"),
        ("reward_select", "proceed"), ("treasure", "open_treasure"),
        ("treasure", "select_treasure_relic"), ("treasure", "skip_treasure_relic")
    };
    var missing = expected.Where(item => !actual.Any(hook => hook.Phase == item.Phase && hook.Action == item.Action)).ToArray();
    if (missing.Length > 0)
        throw new InvalidDataException("missing semantic actions: " + string.Join(", ", missing.Select(x => $"{x.Phase}/{x.Action}")));
    semanticActions.AddRange(actual.Select(hook => new { phase = hook.Phase, action = hook.Action, hook = hook.Type + "." + hook.Method }));

    var exporter = mod.GetType("Sts2HumanRecorder.StateExporter", true)!;
    var enrich = exporter.GetMethod("EnrichFromActionContext", BindingFlags.Static | BindingFlags.Public)!;
    var encoder = mod.GetType("Sts2HumanRecorder.ActionEncoder", true)!;
    var encode = encoder.GetMethod("Encode", BindingFlags.Static | BindingFlags.Public)!;
    var reconcile = mod.GetType("Sts2HumanRecorder.RecorderSession", true)!
        .GetMethod("ReconcileAction", BindingFlags.Static | BindingFlags.NonPublic)!;
    Dictionary<string, object?> BaseObservation(string phase) => new()
    {
        ["phase"] = phase, ["capture_quality"] = "partial",
        ["legal_actions"] = new List<Dictionary<string, object?>>()
    };

    var cardA = new RecorderFakeEntity("CARD.A");
    var cardB = new RecorderFakeEntity("CARD.B");
    var bundleA = new List<object> { cardA };
    var bundleB = new List<object> { cardB };
    var bundleScreen = new RecorderFakeBundleScreen(new List<object> { bundleA, bundleB }, new RecorderFakeBundleNode(bundleB));
    var bundleObs = BaseObservation("bundle_select");
    enrich.Invoke(null, new object?[] { bundleObs, "bundle_select", "select_bundle", bundleScreen, Array.Empty<object?>() });
    var bundleAction = (Dictionary<string, object?>)encode.Invoke(null,
        new object?[] { "select_bundle", bundleScreen, Array.Empty<object?>() })!;
    if (!Equals(((Dictionary<string, object?>)bundleAction["args"]!)["bundle_index"], 1)
        || bundleObs["capture_quality"]?.ToString() != "complete"
        || ((List<Dictionary<string, object?>>)bundleObs["legal_actions"]!).Count != 2)
        throw new InvalidDataException("bundle action/context encoding failed");

    var choiceRelic = new RecorderFakeEntity("RELIC.CHOICE");
    var relicHolder = new RecorderFakeRelicHolder(choiceRelic);
    var relicScreen = new RecorderFakeRelicScreen(new object[] { choiceRelic });
    var relicObs = BaseObservation("relic_select");
    enrich.Invoke(null, new object?[] { relicObs, "relic_select", "choose_relic", relicScreen, Array.Empty<object?>() });
    var relicAction = (Dictionary<string, object?>)encode.Invoke(null,
        new object?[] { "choose_relic", relicScreen, new object?[] { relicHolder } })!;
    if (((Dictionary<string, object?>)relicAction["args"]!)["relic_id"]?.ToString() != "RELIC.CHOICE"
        || relicObs["capture_quality"]?.ToString() != "complete"
        || ((List<Dictionary<string, object?>>)relicObs["legal_actions"]!).Count != 2)
        throw new InvalidDataException("relic selection action/context encoding failed");

    var relic = new RecorderFakeEntity("RELIC.TEST");
    var reward = new RecorderFakeReward(2, "Relic", relic);
    var rewardButton = new RecorderFakeRewardButton(reward, new RecorderFakeRewardSet(new object[] { reward }));
    var rewardObs = BaseObservation("reward_select");
    enrich.Invoke(null, new object?[] { rewardObs, "reward_select", "select_reward", rewardButton, Array.Empty<object?>() });
    var rewardAction = (Dictionary<string, object?>)encode.Invoke(null,
        new object?[] { "select_reward", rewardButton, Array.Empty<object?>() })!;
    var rewardArgs = (Dictionary<string, object?>)rewardAction["args"]!;
    if (!Equals(rewardArgs["reward_index"], 2) || rewardArgs["reward_id"]?.ToString() != "RELIC.TEST"
        || rewardObs["capture_quality"]?.ToString() != "complete")
        throw new InvalidDataException("reward action/context encoding failed");

    var potion = new RecorderFakeEntity("POTION.TEST");
    var potionHolder = new RecorderFakePotionHolder(new RecorderFakeModelNode(potion));
    var potionObs = BaseObservation("potion_manage");
    potionObs["player"] = new Dictionary<string, object?> { ["potions"] = new List<Dictionary<string, object?>>() };
    enrich.Invoke(null, new object?[] { potionObs, "potion_manage", "discard_potion", potionHolder, Array.Empty<object?>() });
    var potionAction = (Dictionary<string, object?>)encode.Invoke(null,
        new object?[] { "discard_potion", potionHolder, Array.Empty<object?>() })!;
    if (((Dictionary<string, object?>)potionAction["args"]!)["potion_id"]?.ToString() != "POTION.TEST"
        || ((List<Dictionary<string, object?>>)((Dictionary<string, object?>)potionObs["player"]!)["potions"]!).Count != 1
        || ((List<Dictionary<string, object?>>)potionObs["legal_actions"]!).All(x => x["action_id"]?.ToString() != "discard_potion")
        || potionObs["capture_quality"]?.ToString() != "complete")
        throw new InvalidDataException("discard potion encoding failed");

    var usePotionObs = BaseObservation("event_choice");
    usePotionObs["player"] = potionObs["player"];
    usePotionObs["foreground_phase"] = "card_reward";
    usePotionObs["run"] = new Dictionary<string, object?>
    {
        ["total_floor"] = 39, ["room_type"] = "Monster", ["room_model_id"] = "ROOM.MONSTER"
    };
    enrich.Invoke(null, new object?[] { usePotionObs, "event_choice", "use_potion", null, new object?[] { potion, null } });
    var usePotionAction = (Dictionary<string, object?>)encode.Invoke(null,
        new object?[] { "use_potion", null, new object?[] { potion, null } })!;
    reconcile.Invoke(null, new object?[] { usePotionObs, usePotionAction });
    var usePotionArgs = (Dictionary<string, object?>)usePotionAction["args"]!;
    if (((List<Dictionary<string, object?>>)usePotionObs["legal_actions"]!).All(x => x["action_id"]?.ToString() != "use_potion")
        || usePotionObs["capture_quality"]?.ToString() != "complete"
        || usePotionArgs["usage_context"]?.ToString() != "card_reward"
        || !Equals(usePotionArgs["origin_floor"], 39)
        || usePotionArgs["origin_room_type"]?.ToString() != "Monster")
        throw new InvalidDataException("non-combat potion context enrichment failed");

    var selectedCard = new RecorderFakeEntity("CARD.SELECTED");
    var pileCard = new RecorderFakeEntity("CARD.PILE");
    var cardSelectScreen = new RecorderFakeCardSelectScreen(Array.Empty<object>(), new[] { selectedCard, pileCard }, new[] { selectedCard });
    var cardSelectObs = BaseObservation("card_select");
    enrich.Invoke(null, new object?[] { cardSelectObs, "card_select", "confirm_card_selection", cardSelectScreen, Array.Empty<object?>() });
    var cardSelect = (Dictionary<string, object?>)cardSelectObs["card_select"]!;
    if (cardSelectObs["capture_quality"]?.ToString() != "complete"
        || ((List<Dictionary<string, object?>>)cardSelect["cards"]!).Count != 2)
        throw new InvalidDataException("combat-pile card selection fallback failed");

    var cancelScreen = new RecorderFakeCardSelectScreen(
        new[] { selectedCard, pileCard }, Array.Empty<object>(), Array.Empty<object>(), 2, 2);
    var cancelObs = BaseObservation("card_select");
    enrich.Invoke(null, new object?[] { cancelObs, "card_select", "skip_card_selection", cancelScreen, Array.Empty<object?>() });
    var cancelSection = (Dictionary<string, object?>)cancelObs["card_select"]!;
    var cancelAction = (Dictionary<string, object?>)encode.Invoke(null,
        new object?[] { "skip_card_selection", cancelScreen, Array.Empty<object?>() })!;
    reconcile.Invoke(null, new object?[] { cancelObs, cancelAction });
    var cancelArgs = (Dictionary<string, object?>)cancelAction["args"]!;
    if (cancelObs["capture_quality"]?.ToString() != "complete"
        || cancelSection["selection_outcome"]?.ToString() != "cancelled"
        || !Equals(cancelSection["min_select"], 2)
        || cancelArgs["selection_outcome"]?.ToString() != "cancelled"
        || ((List<Dictionary<string, object?>>)cancelObs["legal_actions"]!).All(x => x["action_id"]?.ToString() != "skip_card_selection"))
        throw new InvalidDataException("required card-selection cancellation was not captured as complete");

    var shopEntry = new RecorderFakeMerchantEntry(new RecorderFakeEntity("RELIC.SHOP"), 56);
    var shopInventory = new RecorderFakeMerchantInventory(new object[] { shopEntry });
    var shopObs = BaseObservation("shop");
    enrich.Invoke(null, new object?[] { shopObs, "shop", "buy_shop_item", shopEntry, new object?[] { shopInventory, false } });
    if (shopObs["capture_quality"]?.ToString() != "complete"
        || ((List<Dictionary<string, object?>>)shopObs["shop"]!).Count != 1
        || ((List<Dictionary<string, object?>>)shopObs["legal_actions"]!).All(x => x["action_id"]?.ToString() != "buy_shop_item"))
        throw new InvalidDataException("action inventory shop enrichment failed");

    var emptyRewardObs = BaseObservation("reward_select");
    var emptyRewardScreen = new RecorderFakeRewardButton(new RecorderFakeReward(0, "Empty", relic), new RecorderFakeRewardSet(Array.Empty<object>()));
    enrich.Invoke(null, new object?[] { emptyRewardObs, "reward_select", "proceed", emptyRewardScreen, Array.Empty<object?>() });
    if (emptyRewardObs["capture_quality"]?.ToString() != "complete")
        throw new InvalidDataException("terminal empty reward proceed was not complete");

    var treasureHolder = new RecorderFakeTreasureHolder(3, new RecorderFakeModelNode(relic));
    var treasureAction = (Dictionary<string, object?>)encode.Invoke(null,
        new object?[] { "select_treasure_relic", treasureHolder, Array.Empty<object?>() })!;
    var treasureArgs = (Dictionary<string, object?>)treasureAction["args"]!;
    if (!Equals(treasureArgs["relic_index"], 3) || treasureArgs["relic_id"]?.ToString() != "RELIC.TEST")
        throw new InvalidDataException("treasure relic encoding failed");
    actionCoverageSmoke = "PASS";
}
catch (Exception ex)
{
    actionCoverageSmoke = "FAIL: " + (ex.InnerException ?? ex).Message;
    failures.Add("Semantic action coverage smoke: " + actionCoverageSmoke);
}
string writerPerformanceSmoke;
object? writerPerformance = null;
var perfRoot = Path.Combine(Path.GetTempPath(), "sts2-human-recorder-perf-" + Guid.NewGuid().ToString("N"));
var perfPreviousInbox = Environment.GetEnvironmentVariable("STS2_HUMAN_RECORDER_DIR");
try
{
    var inbox = Path.Combine(perfRoot, "inbox");
    Environment.SetEnvironmentVariable("STS2_HUMAN_RECORDER_DIR", inbox);
    var session = mod.GetType("Sts2HumanRecorder.RecorderSession", true)!;
    session.GetMethod("StartNewRun", BindingFlags.Static | BindingFlags.Public)!.Invoke(null, null);
    var recordEvent = session.GetMethod("RecordEngineEvent", BindingFlags.Static | BindingFlags.Public)!;
    var producer = System.Diagnostics.Stopwatch.StartNew();
    for (var i = 0; i < 2000; i++)
        recordEvent.Invoke(null, new object?[] { "writer_stress", new Dictionary<string, object?>
        {
            ["index"] = i, ["payload"] = new string('x', 2048)
        }});
    producer.Stop();
    var writerField = session.GetField("_writer", BindingFlags.Static | BindingFlags.NonPublic)!;
    var writer = writerField.GetValue(null)!;
    var metrics = writer.GetType().GetMethod("Metrics", BindingFlags.Instance | BindingFlags.Public)!.Invoke(writer, null);
    var total = System.Diagnostics.Stopwatch.StartNew();
    session.GetMethod("EndRun", BindingFlags.Static | BindingFlags.Public)!.Invoke(null, new object?[] { "abandoned", false });
    total.Stop();
    var files = Directory.GetFiles(inbox, "*.jsonl");
    if (files.Length != 1) throw new InvalidDataException("writer stress did not seal exactly one file");
    var rows = File.ReadLines(files[0]).Select(line => JsonDocument.Parse(line)).ToList();
    if (rows.Count(x => x.RootElement.GetProperty("record_type").GetString() == "engine_event") != 2000)
        throw new InvalidDataException("writer stress lost records");
    for (var i = 0; i < rows.Count; i++)
        if (rows[i].RootElement.GetProperty("sequence").GetInt64() != i)
            throw new InvalidDataException("writer stress produced a sequence gap");
    foreach (var row in rows) row.Dispose();
    writerPerformance = new
    {
        events = 2000, producer_ms = Math.Round(producer.Elapsed.TotalMilliseconds, 3),
        durable_close_ms = Math.Round(total.Elapsed.TotalMilliseconds, 3), metrics,
        bytes = new FileInfo(files[0]).Length
    };
    writerPerformanceSmoke = "PASS";
}
catch (Exception ex)
{
    writerPerformanceSmoke = "FAIL: " + (ex.InnerException ?? ex).Message;
    failures.Add("Background writer performance smoke: " + writerPerformanceSmoke);
}
finally
{
    Environment.SetEnvironmentVariable("STS2_HUMAN_RECORDER_DIR", perfPreviousInbox);
    if (Directory.Exists(perfRoot)) Directory.Delete(perfRoot, true);
}
Console.WriteLine(JsonSerializer.Serialize(new
{
    status = failures.Count == 0 ? "PASS" : "FAIL", game_assembly = game.GetName().Version?.ToString(),
    mod_assembly = mod.GetName().Version?.ToString(), harmony_patch_smoke = harmonySmoke,
    recorder_lifecycle_smoke = lifecycleSmoke,
    state_matcher_smoke = matcherSmoke,
    victory_mod_isolation_smoke = isolationSmoke,
    combat_schema_smoke = combatSchemaSmoke,
    native_model_state_smoke = nativeModelStateSmoke,
    semantic_action_coverage_smoke = actionCoverageSmoke,
    background_writer_performance_smoke = writerPerformanceSmoke,
    background_writer_performance = writerPerformance,
    semantic_actions = semanticActions,
    attack_intent_api = intentApi,
    hooks = checkedHooks, failures
}, new JsonSerializerOptions { WriteIndented = true }));
return failures.Count == 0 ? 0 : 1;

internal sealed class RecorderFakeEntity(string id)
{
    public string Id { get; } = id;
    public int CanonicalEnergyCost { get; } = 1;
}

internal sealed class RecorderFakeBundleNode(object bundle) { public object Bundle { get; } = bundle; }
internal sealed class RecorderFakeBundleScreen(List<object> bundles, RecorderFakeBundleNode selected)
{
    private readonly List<object> _bundles = bundles;
    private readonly RecorderFakeBundleNode _selectedBundle = selected;
}
internal sealed class RecorderFakeRelicHolder(object model) { public object Model { get; } = model; }
internal sealed class RecorderFakeRelicScreen(IEnumerable<object> relics) { private readonly IEnumerable<object> _relics = relics; }
internal sealed class RecorderFakeReward(int index, string type, object relic)
{
    public int RewardsSetIndex { get; } = index;
    public string RewardType { get; } = type;
    public object Relic { get; } = relic;
    public bool SuccessfullySelected { get; } = false;
}
internal sealed class RecorderFakeRewardSet(IEnumerable<object> rewards) { public IEnumerable<object> Rewards { get; } = rewards; }
internal sealed class RecorderFakeRewardButton(object reward, object set)
{
    public object Reward { get; } = reward;
    private readonly object _rewardsSet = set;
}
internal sealed class RecorderFakeModelNode(object model) { public object Model { get; } = model; }
internal sealed class RecorderFakePotionHolder(object potion) { public object Potion { get; } = potion; }
internal sealed class RecorderFakePile(IEnumerable<object> cards) { public IEnumerable<object> Cards { get; } = cards; }
internal sealed class RecorderFakePrefs(int minSelect = 1, int maxSelect = 1)
{
    public int MinSelect { get; } = minSelect;
    public int MaxSelect { get; } = maxSelect;
}
internal sealed class RecorderFakeCardSelectScreen(IEnumerable<object> cards, IEnumerable<object> pileCards,
    IEnumerable<object> selected, int minSelect = 1, int maxSelect = 1)
{
    private readonly IEnumerable<object> _cards = cards;
    private readonly RecorderFakePile _pile = new(pileCards);
    private readonly IEnumerable<object> _selectedCards = selected;
    private readonly RecorderFakePrefs _prefs = new(minSelect, maxSelect);
}
internal sealed class RecorderFakeMerchantEntry(object model, int cost)
{
    public object Model { get; } = model;
    public int Cost { get; } = cost;
    public bool IsStocked { get; } = true;
    public bool EnoughGold { get; } = true;
}
internal sealed class RecorderFakeMerchantInventory(IEnumerable<object> entries) { public IEnumerable<object> AllEntries { get; } = entries; }
internal sealed class RecorderFakeTreasureHolder(int index, object relic)
{
    public int Index { get; } = index;
    public object Relic { get; } = relic;
}
internal sealed class RecorderFakeSavedProperty<T>(string nameValue, T valueValue)
{
    public string name = nameValue;
    public T value = valueValue;
}
internal sealed class RecorderFakeSavedProperties
{
    public List<RecorderFakeSavedProperty<int>> ints = new();
    public List<RecorderFakeSavedProperty<bool>> bools = new();
    public List<RecorderFakeSavedProperty<string>> strings = new();
    public List<RecorderFakeSavedProperty<int[]>> intArrays = new();
    public List<object> modelIds = new();
    public List<object> cards = new();
    public List<object> cardArrays = new();
}
internal sealed class RecorderFakeEnchantment(string id, int amount)
{
    public string Id { get; } = id;
    public int Amount { get; } = amount;
    public string Status { get; } = "Normal";
    public bool ShowAmount { get; } = true;
    public int DisplayAmount => Amount;
}
internal sealed class RecorderFakeEnergyCost
{
    public int Canonical { get; } = 1;
    public bool CostsX { get; } = false;
    public int CapturedXValue { get; } = 0;
    public int GetResolved() => 1;
}
internal sealed class RecorderFakeCard(string id, object enchantment)
{
    public string Id { get; } = id;
    public int CanonicalEnergyCost { get; } = 1;
    public RecorderFakeEnergyCost EnergyCost { get; } = new();
    public object Enchantment { get; } = enchantment;
    public object? Affliction { get; } = null;
    public int CurrentUpgradeLevel { get; } = 0;
    public int MaxUpgradeLevel { get; } = 1;
    public bool CanPlay() => true;
}
