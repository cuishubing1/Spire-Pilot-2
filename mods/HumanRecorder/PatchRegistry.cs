using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;

namespace Sts2HumanRecorder;

internal static class PatchRegistry
{
    private sealed record Spec(string TypeName, string MethodName, string Phase, string ActionId,
        bool Required = true, bool AwaitBooleanTask = false);
    private sealed record ActionHookState(string Key, ActionCommitToken Commit);
    private static readonly Dictionary<MethodBase, Spec> Actions = new();
    private static readonly List<Dictionary<string, object?>> InstalledHooks = new();
    [ThreadStatic] private static HashSet<string>? ActiveActionHooks;
    private static readonly Spec[] Specs =
    {
        new("MegaCrit.Sts2.Core.Runs.RunManager", "EnterMapCoord", "map_select", "select_map_node"),
        new("MegaCrit.Sts2.Core.Nodes.Combat.NCardPlayQueue", "OnLocalCardPlayed", "combat_play", "play_card"),
        new("MegaCrit.Sts2.Core.Commands.PlayerCmd", "EndTurn", "combat_play", "end_turn"),
        new("MegaCrit.Sts2.Core.Multiplayer.Game.EventSynchronizer", "ChooseLocalOption", "event_choice", "choose_event_option"),
        new("MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer", "ChooseLocalOption", "rest_site", "choose_rest_option"),
        new("MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry", "OnTryPurchaseWrapper", "shop", "buy_shop_item", true, true),
        new("MegaCrit.Sts2.Core.Nodes.Rooms.NMerchantRoom", "HideScreen", "shop", "leave_shop", false),
        new("MegaCrit.Sts2.Core.Multiplayer.Game.OneOffSynchronizer", "DoLocalMerchantCardRemoval", "shop", "remove_card", false),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen", "SelectCard", "card_reward", "choose_card_reward"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen", "OnAlternateRewardSelected", "card_reward", "choose_reward_alternative"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen", "SelectHolder", "card_select", "choose_card", false),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen", "OnSkipButtonReleased", "card_select", "skip", false),
        new("MegaCrit.Sts2.Core.Multiplayer.Game.RewardSynchronizer", "SyncLocalSkippedCard", "card_reward", "skip", false),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckCardSelectScreen", "ConfirmSelection", "card_select", "confirm_card_selection", false),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckCardSelectScreen", "CloseSelection", "card_select", "skip_card_selection", false),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen", "ConfirmSelection", "card_select", "confirm_card_selection"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen", "ConfirmSelection", "card_select", "confirm_card_selection"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen", "CompleteSelection", "card_select", "confirm_card_selection"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen", "ConfirmSelection", "card_select", "confirm_card_selection"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NSimpleCardSelectScreen", "CompleteSelection", "card_select", "confirm_card_selection"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCombatPileCardSelectScreen", "CompleteSelection", "card_select", "confirm_card_selection", false),
        new("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseABundleSelectionScreen", "ConfirmSelection", "bundle_select", "select_bundle"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.NChooseARelicSelection", "SelectHolder", "relic_select", "choose_relic"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.NChooseARelicSelection", "OnSkipButtonReleased", "relic_select", "skip", false),
        new("MegaCrit.Sts2.Core.GameActions.UsePotionAction", ".ctor", "combat_play", "use_potion", false),
        new("MegaCrit.Sts2.Core.Nodes.Potions.NPotionHolder", "DiscardPotion", "potion_manage", "discard_potion"),
        new("MegaCrit.Sts2.Core.Nodes.Rewards.NRewardButton", "OnRelease", "reward_select", "select_reward"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen", "OnProceedButtonPressed", "reward_select", "proceed"),
        new("MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom", "OnChestButtonReleased", "treasure", "open_treasure"),
        new("MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic.NTreasureRoomRelicHolder", "OnRelease", "treasure", "select_treasure_relic"),
        new("MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer", "SkipRelicLocally", "treasure", "skip_treasure_relic")
    };

    public static void Install(Harmony harmony)
    {
        InstalledHooks.Clear();
        var prefix = new HarmonyMethod(typeof(PatchRegistry).GetMethod(nameof(ActionPrefix), BindingFlags.Static | BindingFlags.NonPublic));
        var postfix = new HarmonyMethod(typeof(PatchRegistry).GetMethod(nameof(ActionPostfix), BindingFlags.Static | BindingFlags.NonPublic));
        var asyncPostfix = new HarmonyMethod(typeof(PatchRegistry).GetMethod(nameof(AsyncBooleanActionPostfix), BindingFlags.Static | BindingFlags.NonPublic));
        var finalizer = new HarmonyMethod(typeof(PatchRegistry).GetMethod(nameof(ActionFinalizer), BindingFlags.Static | BindingFlags.NonPublic));
        foreach (var spec in Specs)
        {
            var type = AccessTools.TypeByName(spec.TypeName);
            var methods = type is null ? Array.Empty<MethodBase>() : spec.MethodName == ".ctor"
                ? type.GetConstructors(AccessTools.all).Cast<MethodBase>().ToArray()
                : type.GetMethods(AccessTools.all).Where(x => x.Name == spec.MethodName && x.DeclaringType == type).Cast<MethodBase>().ToArray();
            if (spec.ActionId == "use_potion")
                methods = methods.Where(x => x.GetParameters().FirstOrDefault()?.ParameterType.FullName == "MegaCrit.Sts2.Core.Models.PotionModel").ToArray();
            if (methods.Length == 0)
            {
                InstalledHooks.Add(HookRow(spec.TypeName, spec.MethodName, spec.Phase, spec.ActionId,
                    spec.Required, "missing"));
                if (spec.Required) throw new MissingMethodException(spec.TypeName, spec.MethodName);
                MainFile.Logger.Warn($"Optional recorder hook missing: {spec.TypeName}.{spec.MethodName}");
                continue;
            }
            foreach (var method in methods)
            {
                Actions[method] = spec;
                harmony.Patch(method, prefix: prefix, postfix: spec.AwaitBooleanTask ? asyncPostfix : postfix, finalizer: finalizer);
                InstalledHooks.Add(HookRow(spec.TypeName, spec.MethodName, spec.Phase, spec.ActionId,
                    spec.Required, "installed"));
            }
        }

        Patch(harmony, "MegaCrit.Sts2.Core.Runs.RunManager", "SetUpNewSingleplayer", nameof(NewRunPostfix), postfix: true);
        Patch(harmony, "MegaCrit.Sts2.Core.Runs.RunManager", "SetUpSavedSingleplayer", nameof(ResumeRunPostfix), postfix: true);
        Patch(harmony, "MegaCrit.Sts2.Core.Runs.RunManager", "OnEnded", nameof(RunEndedPostfix), postfix: true);
        Patch(harmony, "MegaCrit.Sts2.Core.Runs.RunManager", "Abandon", nameof(AbandonPostfix), postfix: true);
        Patch(harmony, "MegaCrit.Sts2.Core.Commands.CreatureCmd", "Heal", nameof(HealPrefix), postfix: false);
        var relicType = AccessTools.TypeByName("MegaCrit.Sts2.Core.Models.RelicModel")
            ?? throw new TypeLoadException("MegaCrit.Sts2.Core.Models.RelicModel");
        var relicFlash = new HarmonyMethod(typeof(PatchRegistry).GetMethod(nameof(RelicFlashPrefix), BindingFlags.Static | BindingFlags.NonPublic));
        foreach (var method in relicType.GetMethods(AccessTools.all).Where(method => method.Name == "Flash" && method.DeclaringType == relicType))
        {
            harmony.Patch(method, prefix: relicFlash);
            InstalledHooks.Add(HookRow(relicType.FullName ?? relicType.Name, method.Name, "audit", "relic_trigger_observed", true, "installed"));
        }
    }

    public static List<Dictionary<string, object?>> HookManifest()
    {
        lock (InstalledHooks)
            return InstalledHooks.Select(row => new Dictionary<string, object?>(row, StringComparer.Ordinal)).ToList();
    }

    private static void Patch(Harmony harmony, string typeName, string methodName, string patchName, bool postfix)
    {
        var method = AccessTools.Method(AccessTools.TypeByName(typeName), methodName) ?? throw new MissingMethodException(typeName, methodName);
        var patch = new HarmonyMethod(typeof(PatchRegistry).GetMethod(patchName, BindingFlags.Static | BindingFlags.NonPublic));
        harmony.Patch(method, prefix: postfix ? null : patch, postfix: postfix ? patch : null);
        InstalledHooks.Add(HookRow(typeName, methodName, "lifecycle", patchName, true, "installed"));
    }

    private static void ActionPrefix(MethodBase __originalMethod, object? __instance, object[] __args, out ActionHookState? __state)
    {
        __state = null;
        try
        {
            if (Actions.TryGetValue(__originalMethod, out var spec))
            {
                var phase = spec.Phase;
                if (spec.ActionId == "use_potion")
                {
                    phase = ResolvePotionPhase();
                }
                else if (spec.ActionId == "discard_potion")
                {
                    var combatManager = ReflectionUtil.Get(AccessTools.TypeByName("MegaCrit.Sts2.Core.Combat.CombatManager"), "Instance");
                    if (ReflectionUtil.Bool(ReflectionUtil.Get(combatManager, "IsInProgress"))) phase = "combat_play";
                }
                var key = phase + "/" + spec.ActionId;
                ActiveActionHooks ??= new HashSet<string>(StringComparer.Ordinal);
                if (!ActiveActionHooks.Add(key)) return;
                var commit = RecorderSession.RecordAction(phase, spec.ActionId, __instance, __args);
                if (commit is null)
                {
                    ActiveActionHooks.Remove(key);
                    return;
                }
                __state = new ActionHookState(key, commit);
            }
        }
        catch (Exception ex)
        {
            ActiveActionHooks?.Clear();
            MainFile.Logger.Error($"Recorder action hook failed: {ex}");
        }
    }

    private static string ResolvePotionPhase()
    {
        var combatManager = ReflectionUtil.Get(AccessTools.TypeByName("MegaCrit.Sts2.Core.Combat.CombatManager"), "Instance");
        if (ReflectionUtil.Bool(ReflectionUtil.Get(combatManager, "IsInProgress"))) return "combat_play";
        var runManager = ReflectionUtil.Get(AccessTools.TypeByName("MegaCrit.Sts2.Core.Runs.RunManager"), "Instance");
        var state = ReflectionUtil.Get(runManager, "State");
        var roomType = ReflectionUtil.Get(ReflectionUtil.Get(state, "CurrentRoom"), "RoomType")?.ToString() ?? "";
        if (roomType.Contains("Event", StringComparison.OrdinalIgnoreCase)) return "event_choice";
        if (roomType.Contains("Shop", StringComparison.OrdinalIgnoreCase)
            || roomType.Contains("Merchant", StringComparison.OrdinalIgnoreCase)) return "shop";
        if (roomType.Contains("Rest", StringComparison.OrdinalIgnoreCase)) return "rest_site";
        if (roomType.Contains("Treasure", StringComparison.OrdinalIgnoreCase)) return "treasure";
        return "potion_manage";
    }

    private static void ActionPostfix(ActionHookState? __state)
    {
        if (__state is null) return;
        __state.Commit.Complete("method_returned");
        ActiveActionHooks?.Remove(__state.Key);
    }

    private static void AsyncBooleanActionPostfix(ActionHookState? __state, Task<bool> __result)
    {
        if (__state is null) return;
        ActiveActionHooks?.Remove(__state.Key);
        _ = __result.ContinueWith(task =>
        {
            var committed = task.Status == TaskStatus.RanToCompletion && task.Result;
            __state.Commit.Complete(committed ? "committed" : "failed");
        }, TaskScheduler.Default);
    }

    private static Exception? ActionFinalizer(ActionHookState? __state, Exception? __exception)
    {
        if (__state is not null)
        {
            if (__exception is not null) __state.Commit.Complete("failed");
            ActiveActionHooks?.Remove(__state.Key);
        }
        // Never swallow exceptions from the game or another mod. The finalizer exists
        // only to ensure recorder recursion state is cleaned after an exceptional call.
        return __exception;
    }

    private static void NewRunPostfix() => SafeLifecycle("new run", RecorderSession.StartNewRun);
    private static void ResumeRunPostfix() => SafeLifecycle("resume run", RecorderSession.ResumeRun);
    private static void RunEndedPostfix(object? __result)
    {
        var won = ReflectionUtil.Get(__result, "Victory", "Won", "IsVictory");
        SafeLifecycle("run end", () => RecorderSession.EndRun("game_ended", won is null ? null : ReflectionUtil.Bool(won)));
    }
    private static void AbandonPostfix() => SafeLifecycle("abandon", () => RecorderSession.EndRun("abandoned", false));

    private static void SafeLifecycle(string name, Action action)
    {
        try { action(); }
        catch (Exception ex)
        {
            MainFile.Logger.Error($"Recorder {name} hook failed without affecting gameplay: {ex}");
            RecorderSession.ReportFatalInitialization(ex);
        }
    }

    private static Dictionary<string, object?> HookRow(string type, string method, string phase,
        string action, bool required, string status) => new()
    {
        ["type"] = type, ["method"] = method, ["phase"] = phase, ["action_id"] = action,
        ["required"] = required, ["status"] = status
    };

    private static void HealPrefix(object[] __args)
    {
        try
        {
            var creature = __args.FirstOrDefault();
            var monster = ReflectionUtil.Get(creature, "Monster");
            var player = ReflectionUtil.Get(creature, "Player");
            var playerModel = ReflectionUtil.Get(player, "Character", "CharacterModel", "Model");
            RecorderSession.RecordEngineEvent("heal_requested", new Dictionary<string, object?>
            {
                ["target_kind"] = monster is not null ? "monster" : player is not null ? "player" : "creature",
                ["target_id"] = ReflectionUtil.Id(monster) ?? ReflectionUtil.Id(playerModel) ?? ReflectionUtil.Id(creature),
                ["target_combat_id"] = ReflectionUtil.Get(creature, "CombatId")?.ToString(),
                ["hp_before"] = ReflectionUtil.Int(ReflectionUtil.Get(creature, "CurrentHp")),
                ["max_hp"] = ReflectionUtil.Int(ReflectionUtil.Get(creature, "MaxHp")),
                ["requested_amount"] = __args.Skip(1).FirstOrDefault(),
                ["play_animation"] = ReflectionUtil.Bool(__args.Skip(2).FirstOrDefault())
            });
        }
        catch (Exception ex) { MainFile.Logger.Error($"Recorder heal event hook failed: {ex}"); }
    }

    private static void RelicFlashPrefix(object __instance, object[] __args)
    {
        try
        {
            var targets = __args.SelectMany(ReflectionUtil.Items).Select(target => new Dictionary<string, object?>
            {
                ["target_id"] = ReflectionUtil.Id(ReflectionUtil.Get(target, "Monster")) ?? ReflectionUtil.Id(target),
                ["target_combat_id"] = ReflectionUtil.Get(target, "CombatId")?.ToString()
            }).ToList();
            RecorderSession.RecordEngineEvent("relic_trigger_observed", new Dictionary<string, object?>
            {
                ["source_id"] = ReflectionUtil.Id(__instance),
                ["source_object_ref"] = RuntimeHelpers.GetHashCode(__instance).ToString("x8"),
                ["signal"] = "RelicModel.Flash",
                ["targets"] = targets,
                ["state"] = NativeModelState.Relic(__instance, true)
            });
        }
        catch (Exception ex) { MainFile.Logger.Error($"Recorder relic trigger event hook failed: {ex}"); }
    }
}
