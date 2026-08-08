using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using Godot;
using HarmonyLib;

namespace Sts2HumanRecorder;

internal static class StateExporter
{
    public static Dictionary<string, object?> CaptureRunContext()
    {
        var result = new Dictionary<string, object?>
        {
            ["capture_quality"] = "complete"
        };
        try
        {
            var managerType = AccessTools.TypeByName("MegaCrit.Sts2.Core.Runs.RunManager");
            var manager = ReflectionUtil.Get(managerType, "Instance");
            var state = ReflectionUtil.Get(manager, "State");
            if (state is null)
            {
                result["capture_quality"] = "partial";
                result["capture_errors"] = new[] { "run_state_unavailable" };
                return result;
            }

            var rng = ReflectionUtil.Get(state, "Rng");
            var players = ReflectionUtil.Items(ReflectionUtil.Get(state, "Players")).ToList();
            result["seed"] = ReflectionUtil.Get(rng, "StringSeed")?.ToString();
            result["seed_numeric"] = ReflectionUtil.Get(rng, "Seed");
            result["game_mode"] = ReflectionUtil.Get(state, "GameMode")?.ToString();
            result["ascension"] = ReflectionUtil.Int(ReflectionUtil.Get(state, "AscensionLevel"));
            result["character_ids"] = players
                .Select(player => ReflectionUtil.Id(ReflectionUtil.Get(player, "Character")))
                .Where(id => !string.IsNullOrWhiteSpace(id)).ToList();
            result["act_ids"] = ReflectionUtil.Items(ReflectionUtil.Get(state, "Acts"))
                .Select(ReflectionUtil.Id).Where(id => !string.IsNullOrWhiteSpace(id)).ToList();
            result["modifier_ids"] = ReflectionUtil.Items(ReflectionUtil.Get(state, "Modifiers"))
                .Select(ReflectionUtil.Id).Where(id => !string.IsNullOrWhiteSpace(id)).ToList();
            result["badge_ids"] = ReflectionUtil.Items(ReflectionUtil.Get(state, "BadgeModels"))
                .Select(ReflectionUtil.Id).Where(id => !string.IsNullOrWhiteSpace(id)).ToList();
            result["should_save"] = ReflectionUtil.Bool(ReflectionUtil.Get(manager, "ShouldSave"));
            result["daily_time"] = ReflectionUtil.Get(manager, "DailyTime")?.ToString();
        }
        catch (Exception ex)
        {
            result["capture_quality"] = "partial";
            result["capture_errors"] = new[] { ex.GetType().Name + ": " + ex.Message };
        }
        return result;
    }

    public static Dictionary<string, object?> Capture(string phaseHint)
    {
        var result = new Dictionary<string, object?>
        {
            ["phase"] = phaseHint,
            ["capture_quality"] = "complete"
        };
        try
        {
            var managerType = AccessTools.TypeByName("MegaCrit.Sts2.Core.Runs.RunManager");
            var manager = ReflectionUtil.Get(managerType, "Instance");
            var state = ReflectionUtil.Get(manager, "State");
            if (state is null)
            {
                result["capture_quality"] = "partial";
                result["capture_errors"] = new[] { "run_state_unavailable" };
                return result;
            }

            result["run"] = RunSummary(state);
            var player = ReflectionUtil.Items(ReflectionUtil.Get(state, "Players")).FirstOrDefault();
            if (player is not null) result["player"] = PlayerSummary(player);
            if (phaseHint is "combat_play" or "card_select" || (phaseHint == "potion_manage" && IsCombatInProgress()))
            {
                var combat = CombatSummary(player);
                result["combat"] = combat;
                if (!ReflectionUtil.Bool(combat.GetValueOrDefault("intent_capture_complete"), true))
                {
                    result["capture_quality"] = "partial";
                    result["capture_errors"] = new[] { "visible_attack_intent_damage_unavailable" };
                }
            }
            if (phaseHint == "map_select") result["visible_map"] = MapSummary(state);
            if (phaseHint == "event_choice") result["event"] = EventSummary(state);
            if (phaseHint == "rest_site") result["rest_site"] = RestSummary(state);
            if (phaseHint == "shop") result["shop"] = ShopSummary(state, player);
            result["legal_actions"] = LegalActions(phaseHint, result, state, player);
            if (((List<Dictionary<string, object?>>)result["legal_actions"]!).Count == 0 && phaseHint != "game_over")
                result["capture_quality"] = "partial";
            ApplyNativeStateQuality(result);
        }
        catch (Exception ex)
        {
            result["capture_quality"] = "partial";
            result["capture_errors"] = new[] { ex.GetType().Name + ": " + ex.Message };
        }
        return result;
    }

    public static Dictionary<string, object?> CaptureAuditState(string phaseHint)
    {
        var result = new Dictionary<string, object?>
        {
            ["schema_version"] = NativeModelState.SchemaVersion,
            ["projection_version"] = ContentStateRegistry.Version,
            ["phase"] = phaseHint,
            ["capture_quality"] = "complete"
        };
        try
        {
            var manager = ReflectionUtil.Get(AccessTools.TypeByName("MegaCrit.Sts2.Core.Runs.RunManager"), "Instance");
            var state = ReflectionUtil.Get(manager, "State");
            var player = ReflectionUtil.Items(ReflectionUtil.Get(state, "Players")).FirstOrDefault();
            if (player is null) return result;
            result["player"] = new Dictionary<string, object?>
            {
                ["deck"] = AuditCards(ReflectionUtil.Get(ReflectionUtil.Get(player, "Deck"), "Cards")),
                ["relics"] = ReflectionUtil.Items(ReflectionUtil.Get(player, "Relics"))
                    .Select(relic => NativeModelState.Relic(relic, true)).ToList()
            };
            if (IsCombatInProgress())
            {
                var pcs = ReflectionUtil.Get(player, "PlayerCombatState");
                result["combat"] = new Dictionary<string, object?>
                {
                    ["hand"] = AuditCards(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "Hand"), "Cards")),
                    ["draw_pile_ordered"] = AuditCards(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "DrawPile"), "Cards")),
                    ["discard_pile_ordered"] = AuditCards(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "DiscardPile"), "Cards")),
                    ["exhaust_pile_ordered"] = AuditCards(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "ExhaustPile"), "Cards"))
                };
            }
            ApplyNativeStateQuality(result);
        }
        catch (Exception ex)
        {
            result["capture_quality"] = "partial";
            result["capture_errors"] = new[] { ex.GetType().Name + ": " + ex.Message };
        }
        return result;
    }

    public static void EnrichFromActionContext(Dictionary<string, object?> observation, string phase, string actionId,
        object? instance, object?[] args)
    {
        if (actionId == "use_potion") { EnrichUsePotion(observation, args); return; }
        if (instance is null) return;
        if (actionId == "discard_potion") EnrichDiscardPotion(observation, instance);
        if (phase == "shop" && actionId == "buy_shop_item") { EnrichShopPurchase(observation, args); return; }
        if (phase == "bundle_select") { EnrichBundle(observation, instance); return; }
        if (phase == "relic_select") { EnrichRelicSelection(observation, instance); return; }
        if (phase == "reward_select") { EnrichRewards(observation, instance, actionId); return; }
        if (phase == "treasure") { EnrichTreasure(observation, instance, actionId); return; }
        if (phase is not ("card_reward" or "card_select")) return;
        try
        {
            object? rawCards = null;
            var alternatives = new List<Dictionary<string, object?>>();
            var canSkip = false;
            if (phase == "card_reward")
            {
                rawCards = ReflectionUtil.Get(instance, "_options");
                rawCards = ReflectionUtil.Items(rawCards).Select(x => ReflectionUtil.Get(x, "Card") ?? x).ToList();
                alternatives = ReflectionUtil.Items(ReflectionUtil.Get(instance, "_extraOptions"))
                    .Select((x, i) => new Dictionary<string, object?>
                    {
                        ["index"] = i, ["id"] = ReflectionUtil.Id(x), ["type"] = x.GetType().Name
                    }).ToList();
                canSkip = true;
            }
            else
            {
                rawCards = ReflectionUtil.Get(instance, "_cards");
                if (!ReflectionUtil.Items(rawCards).Any())
                    rawCards = ReflectionUtil.Get(ReflectionUtil.Get(instance, "_pile"), "Cards");
                if (!ReflectionUtil.Items(rawCards).Any())
                    rawCards = ReflectionUtil.Get(instance, "_selectedCards");
                canSkip = ReflectionUtil.Bool(ReflectionUtil.Get(instance, "_canSkip"))
                    || ReflectionUtil.Int(ReflectionUtil.Get(ReflectionUtil.Get(instance, "_prefs"), "MinSelect"), 1) == 0;
            }
            var cards = CardList(rawCards, phase);
            var selected = CardList(ReflectionUtil.Get(instance, "_selectedCards"), "selected");
            var isCancellation = actionId is "skip_card_selection" or "skip";
            observation[phase] = new Dictionary<string, object?>
            {
                ["cards"] = cards, ["selected_cards"] = selected,
                ["alternatives"] = alternatives, ["can_skip"] = canSkip,
                ["selection_outcome"] = isCancellation ? "cancelled" :
                    actionId == "confirm_card_selection" ? "confirmed" : "selected",
                ["cancellation_observed"] = isCancellation,
                ["min_select"] = ReflectionUtil.Int(ReflectionUtil.Get(ReflectionUtil.Get(instance, "_prefs"), "MinSelect"), phase == "card_reward" ? 1 : 0),
                ["max_select"] = ReflectionUtil.Int(ReflectionUtil.Get(ReflectionUtil.Get(instance, "_prefs"), "MaxSelect"), Math.Max(1, selected.Count))
            };
            List<Dictionary<string, object?>> legal;
            if (actionId is "confirm_card_selection" or "skip_card_selection")
                legal = new() { Action(actionId, new() { ["selected_card_ids"] = selected.Select(x => x["id"]).ToList() }) };
            else
            {
                legal = cards.Select(x => Action(phase == "card_reward" ? "choose_card_reward" : "choose_card",
                    new() { ["card_instance_id"] = x["instance_id"], ["card_id"] = x["id"] })).ToList();
                foreach (var alternative in alternatives)
                    legal.Add(Action("choose_reward_alternative", new() { ["index"] = alternative["index"], ["id"] = alternative["id"] }));
                if (canSkip) legal.Add(Action("skip", new()));
            }
            observation["legal_actions"] = legal;
            var prefs = ReflectionUtil.Get(instance, "_prefs");
            var minSelect = ReflectionUtil.Int(ReflectionUtil.Get(prefs, "MinSelect"), actionId == "skip_card_selection" ? 0 : 1);
            var maxSelect = ReflectionUtil.Int(ReflectionUtil.Get(prefs, "MaxSelect"), Math.Max(minSelect, selected.Count));
            // A visible cancel/return action is a complete observation even when
            // the selection preferences require N cards. The minimum applies to
            // confirmation, not to abandoning the child selection screen.
            var selectionValid = actionId != "confirm_card_selection"
                || (selected.Count >= minSelect && selected.Count <= Math.Max(minSelect, maxSelect));
            var contextComplete = cards.Count > 0 && selectionValid;
            observation["capture_quality"] = legal.Count > 0 && contextComplete ? "complete" : "partial";
        }
        catch (Exception ex)
        {
            observation["capture_quality"] = "partial";
            observation["capture_errors"] = new[] { "selection_context: " + ex.Message };
        }
    }

    private static void EnrichUsePotion(Dictionary<string, object?> observation, object?[] args)
    {
        try
        {
            EnrichVisiblePotionForeground(observation);
            var potion = args.FirstOrDefault();
            if (potion is null) return;
            var player = observation.GetValueOrDefault("player") as Dictionary<string, object?>;
            var potions = player?.GetValueOrDefault("potions") as List<Dictionary<string, object?>>;
            if (potions is null) return;
            var objectId = RuntimeHelpers.GetHashCode(potion).ToString("x8");
            var row = potions.FirstOrDefault(x => x.GetValueOrDefault("instance_id")?.ToString()
                ?.EndsWith(":" + objectId, StringComparison.Ordinal) == true);
            if (row is null) return;
            var actionArgs = new Dictionary<string, object?>
            {
                ["potion_instance_id"] = row["instance_id"], ["potion_id"] = row["id"]
            };
            var target = args.Skip(1).FirstOrDefault();
            if (target is not null)
            {
                actionArgs["target_combat_id"] = ReflectionUtil.Get(target, "CombatId")?.ToString();
                actionArgs["target_id"] = ReflectionUtil.Id(ReflectionUtil.Get(target, "Monster")) ?? ReflectionUtil.Id(target);
            }
            var legal = observation.GetValueOrDefault("legal_actions") as List<Dictionary<string, object?>> ?? new();
            if (!legal.Any(x => x.GetValueOrDefault("action_id")?.ToString() == "use_potion"
                && (x.GetValueOrDefault("args") as Dictionary<string, object?>)?.GetValueOrDefault("potion_instance_id")?.ToString()
                    == row.GetValueOrDefault("instance_id")?.ToString()))
                legal.Add(Action("use_potion", actionArgs));
            observation["legal_actions"] = legal;
            PromotePotionContextQuality(observation);
        }
        catch (Exception ex)
        {
            observation["capture_quality"] = "partial";
            observation["capture_errors"] = new[] { "use_potion_context: " + ex.Message };
        }
    }

    private static void EnrichVisiblePotionForeground(Dictionary<string, object?> observation)
    {
        var cardRewardScreen = FindVisibleNode("NCardRewardSelectionScreen");
        if (cardRewardScreen is not null)
        {
            var cards = CardList(ReflectionUtil.Items(ReflectionUtil.Get(cardRewardScreen, "_options"))
                .Select(x => ReflectionUtil.Get(x, "Card") ?? x), "card_reward");
            var alternatives = ReflectionUtil.Items(ReflectionUtil.Get(cardRewardScreen, "_extraOptions"))
                .Select((x, i) => new Dictionary<string, object?>
                {
                    ["index"] = i, ["id"] = ReflectionUtil.Id(x), ["type"] = x.GetType().Name
                }).ToList();
            if (cards.Count > 0 || alternatives.Count > 0)
            {
                observation["foreground_phase"] = "card_reward";
                observation["card_reward"] = new Dictionary<string, object?>
                {
                    ["cards"] = cards, ["selected_cards"] = new List<Dictionary<string, object?>>(),
                    ["alternatives"] = alternatives, ["can_skip"] = true,
                    ["selection_outcome"] = "pending", ["cancellation_observed"] = false,
                    ["min_select"] = 1, ["max_select"] = 1
                };
                var legal = observation.GetValueOrDefault("legal_actions") as List<Dictionary<string, object?>> ?? new();
                foreach (var card in cards) legal.Add(Action("choose_card_reward", new()
                {
                    ["card_instance_id"] = card["instance_id"], ["card_id"] = card["id"]
                }));
                foreach (var alternative in alternatives) legal.Add(Action("choose_reward_alternative", new()
                {
                    ["index"] = alternative["index"], ["id"] = alternative["id"]
                }));
                legal.Add(Action("skip", new()));
                observation["legal_actions"] = legal;
                return;
            }
        }

        var rewardsScreen = FindVisibleNode("NRewardsScreen");
        if (rewardsScreen is not null)
        {
            EnrichRewards(observation, rewardsScreen, "proceed");
            if (observation.GetValueOrDefault("reward_select") is not null)
                observation["foreground_phase"] = "reward_select";
        }
    }

    private static Node? FindVisibleNode(string typeName)
    {
        if (System.Environment.GetEnvironmentVariable("STS2_HUMAN_RECORDER_DISABLE_GODOT_LOOKUP") == "1") return null;
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree || tree.Root is null) return null;
            var pending = new Queue<Node>();
            pending.Enqueue(tree.Root);
            var visited = 0;
            while (pending.Count > 0 && visited++ < 20000)
            {
                var node = pending.Dequeue();
                if (node.GetType().Name == typeName
                    && ReflectionUtil.Bool(ReflectionUtil.Get(node, "Visible"), true)) return node;
                foreach (var child in node.GetChildren()) pending.Enqueue(child);
            }
        }
        catch { }
        return null;
    }

    private static void EnrichDiscardPotion(Dictionary<string, object?> observation, object instance)
    {
        try
        {
            var potionNode = ReflectionUtil.Get(instance, "Potion");
            var potion = ReflectionUtil.Get(potionNode, "Model", "PotionModel", "Potion") ?? potionNode;
            if (potion is null) return;
            var player = observation.GetValueOrDefault("player") as Dictionary<string, object?>;
            var potions = player?.GetValueOrDefault("potions") as List<Dictionary<string, object?>>;
            if (potions is null) return;
            var objectId = RuntimeHelpers.GetHashCode(potion).ToString("x8");
            var row = potions.FirstOrDefault(x => x.GetValueOrDefault("instance_id")?.ToString()
                ?.EndsWith(":" + objectId, StringComparison.Ordinal) == true);
            if (row is null)
            {
                row = ModelList(new[] { potion }, "potion").FirstOrDefault();
                if (row is null) return;
                var index = potions.Count;
                row["index"] = index;
                row["instance_id"] = $"potion:{index}:{objectId}";
                potions.Add(row);
            }
            var legal = observation.GetValueOrDefault("legal_actions") as List<Dictionary<string, object?>> ?? new();
            if (!legal.Any(x => x.GetValueOrDefault("action_id")?.ToString() == "discard_potion"
                && (x.GetValueOrDefault("args") as Dictionary<string, object?>)?.GetValueOrDefault("potion_instance_id")?.ToString()
                    == row.GetValueOrDefault("instance_id")?.ToString()))
            {
                legal.Add(Action("discard_potion", new()
                {
                    ["potion_instance_id"] = row["instance_id"], ["potion_id"] = row["id"]
                }));
            }
            observation["legal_actions"] = legal;
            PromotePotionContextQuality(observation);
        }
        catch (Exception ex)
        {
            observation["capture_quality"] = "partial";
            observation["capture_errors"] = new[] { "discard_potion_context: " + ex.Message };
        }
    }

    private static void PromotePotionContextQuality(Dictionary<string, object?> observation)
    {
        if (!observation.ContainsKey("capture_errors")
            && (observation.GetValueOrDefault("legal_actions") as List<Dictionary<string, object?>>)?.Count > 0)
            observation["capture_quality"] = "complete";
    }

    private static void EnrichBundle(Dictionary<string, object?> observation, object instance)
    {
        try
        {
            var bundles = ReflectionUtil.Items(ReflectionUtil.Get(instance, "_bundles")).Select((bundle, index) =>
                new Dictionary<string, object?>
                {
                    ["index"] = index,
                    ["cards"] = CardList(bundle, $"bundle:{index}")
                }).ToList();
            observation["bundle_select"] = new Dictionary<string, object?> { ["bundles"] = bundles };
            observation["legal_actions"] = bundles.Select(bundle => Action("select_bundle", new()
            {
                ["bundle_index"] = bundle["index"],
                ["card_ids"] = ((List<Dictionary<string, object?>>)bundle["cards"]!).Select(card => card["id"]).ToList()
            })).ToList();
            observation["capture_quality"] = bundles.Count > 0
                && bundles.All(bundle => ((List<Dictionary<string, object?>>)bundle["cards"]!).Count > 0) ? "complete" : "partial";
        }
        catch (Exception ex)
        {
            observation["capture_quality"] = "partial";
            observation["capture_errors"] = new[] { "bundle_context: " + ex.Message };
        }
    }

    private static void EnrichRewards(Dictionary<string, object?> observation, object instance, string actionId)
    {
        try
        {
            var screen = FindAncestorWithMember(instance, "_rewardsSet") ?? instance;
            var rewardsSet = ReflectionUtil.Get(screen, "_rewardsSet");
            var rewards = RewardRows(ReflectionUtil.Get(rewardsSet, "Rewards"));
            if (rewards.Count == 0 && actionId == "select_reward")
            {
                var clicked = ReflectionUtil.Get(instance, "Reward");
                if (clicked is not null) rewards = RewardRows(new[] { clicked });
            }
            var skipDisallowed = ReflectionUtil.Bool(ReflectionUtil.Get(screen, "_skipDisallowed", "DisallowSkipping"));
            var allSelected = rewards.Count > 0 && rewards.All(row => ReflectionUtil.Bool(row.GetValueOrDefault("selected")));
            observation["reward_select"] = new Dictionary<string, object?>
            {
                ["rewards"] = rewards, ["skip_disallowed"] = skipDisallowed,
                ["is_terminal"] = ReflectionUtil.Bool(ReflectionUtil.Get(screen, "_isTerminal"))
            };
            var legal = rewards.Where(row => !ReflectionUtil.Bool(row.GetValueOrDefault("selected"))).Select(row => Action("select_reward", new()
            {
                ["reward_index"] = row["index"], ["reward_type"] = row["type"], ["reward_id"] = row["id"]
            })).ToList();
            if (!skipDisallowed || allSelected) legal.Add(Action("proceed", new()));
            observation["legal_actions"] = legal;
            observation["capture_quality"] = legal.Count > 0
                && (rewards.Count > 0 || actionId == "proceed") ? "complete" : "partial";
        }
        catch (Exception ex)
        {
            observation["capture_quality"] = "partial";
            observation["capture_errors"] = new[] { "reward_context: " + ex.Message };
        }
    }

    private static void EnrichRelicSelection(Dictionary<string, object?> observation, object instance)
    {
        try
        {
            var relics = RelicList(ReflectionUtil.Get(instance, "_relics"), "relic_choice");
            observation["relic_select"] = new Dictionary<string, object?> { ["relics"] = relics, ["can_skip"] = true };
            var legal = relics.Select(relic => Action("choose_relic", new()
            {
                ["relic_index"] = relic["index"], ["relic_id"] = relic["id"],
                ["relic_instance_id"] = relic["instance_id"]
            })).ToList();
            legal.Add(Action("skip", new()));
            observation["legal_actions"] = legal;
            observation["capture_quality"] = relics.Count > 0 ? "complete" : "partial";
        }
        catch (Exception ex)
        {
            observation["capture_quality"] = "partial";
            observation["capture_errors"] = new[] { "relic_selection_context: " + ex.Message };
        }
    }

    private static void EnrichTreasure(Dictionary<string, object?> observation, object instance, string actionId)
    {
        try
        {
            var manager = ReflectionUtil.Get(AccessTools.TypeByName("MegaCrit.Sts2.Core.Runs.RunManager"), "Instance");
            var synchronizer = ReflectionUtil.Get(manager, "TreasureRoomRelicSynchronizer");
            // CurrentRelics is populated before the chest button is released. It is engine-internal
            // hidden information at the open decision, so never expose it until after that action.
            var relics = actionId == "open_treasure"
                ? new List<Dictionary<string, object?>>()
                : RelicList(ReflectionUtil.Get(synchronizer, "CurrentRelics"), "treasure_relic");
            if (relics.Count == 0 && actionId == "select_treasure_relic")
            {
                var node = ReflectionUtil.Get(instance, "Relic");
                var relic = ReflectionUtil.Get(node, "Model", "RelicModel", "Relic") ?? node;
                if (relic is not null) relics = RelicList(new[] { relic }, "treasure_relic");
            }
            observation["treasure"] = new Dictionary<string, object?>
            {
                ["relics"] = relics,
                ["opened"] = relics.Count > 0,
                ["room_id"] = ReflectionUtil.Id(ReflectionUtil.Get(ReflectionUtil.Get(manager, "State"), "CurrentRoom"))
            };
            var legal = new List<Dictionary<string, object?>>();
            if (relics.Count == 0) legal.Add(Action("open_treasure", new()));
            else
            {
                foreach (var relic in relics) legal.Add(Action("select_treasure_relic", new()
                {
                    ["relic_index"] = relic["index"], ["relic_id"] = relic["id"]
                }));
                legal.Add(Action("skip_treasure_relic", new()));
            }
            observation["legal_actions"] = legal;
            observation["capture_quality"] = legal.Count > 0 ? "complete" : "partial";
        }
        catch (Exception ex)
        {
            observation["capture_quality"] = "partial";
            observation["capture_errors"] = new[] { "treasure_context: " + ex.Message };
        }
    }

    private static void EnrichShopPurchase(Dictionary<string, object?> observation, object?[] args)
    {
        try
        {
            var inventory = args.FirstOrDefault();
            var entries = ShopRows(inventory);
            if (entries.Count == 0) return;
            observation["shop"] = entries;
            var legal = entries.Where(x => ReflectionUtil.Bool(x["stocked"]) && ReflectionUtil.Bool(x["affordable"]))
                .Select(entry => Action(entry["kind"]?.ToString()?.Contains("CardRemoval", StringComparison.OrdinalIgnoreCase) == true
                    ? "remove_card" : "buy_shop_item", new()
                {
                    ["index"] = entry["index"], ["id"] = entry["id"], ["cost"] = entry["cost"]
                })).ToList();
            legal.Add(Action("leave_shop", new()));
            observation["legal_actions"] = legal;
            observation["capture_quality"] = "complete";
        }
        catch (Exception ex)
        {
            observation["capture_quality"] = "partial";
            observation["capture_errors"] = new[] { "shop_purchase_context: " + ex.Message };
        }
    }

    private static Dictionary<string, object?> RunSummary(object state) => new()
    {
        ["act"] = ReflectionUtil.Int(ReflectionUtil.Get(state, "CurrentActIndex")) + 1,
        ["act_id"] = ReflectionUtil.Id(ReflectionUtil.Get(state, "Act")),
        ["act_floor"] = ReflectionUtil.Int(ReflectionUtil.Get(state, "ActFloor")),
        ["total_floor"] = ReflectionUtil.Int(ReflectionUtil.Get(state, "TotalFloor")),
        ["ascension"] = ReflectionUtil.Int(ReflectionUtil.Get(state, "AscensionLevel")),
        ["room_type"] = ReflectionUtil.Get(ReflectionUtil.Get(state, "CurrentRoom"), "RoomType")?.ToString(),
        ["room_model_id"] = ReflectionUtil.Id(ReflectionUtil.Get(state, "CurrentRoom")),
        ["map_coord"] = Coord(ReflectionUtil.Get(state, "CurrentMapCoord")),
        ["is_game_over"] = ReflectionUtil.Bool(ReflectionUtil.Get(state, "IsGameOver"))
    };

    private static Dictionary<string, object?> PlayerSummary(object player)
    {
        var creature = ReflectionUtil.Get(player, "Creature");
        var character = ReflectionUtil.Get(player, "Character");
        CardIdentity.BeginDeckSnapshot();
        List<Dictionary<string, object?>> deck;
        try { deck = CardList(ReflectionUtil.Get(ReflectionUtil.Get(player, "Deck"), "Cards"), "deck"); }
        finally { CardIdentity.EndDeckSnapshot(); }
        var result = new Dictionary<string, object?>
        {
            ["character_id"] = ReflectionUtil.Id(character),
            ["hp"] = ReflectionUtil.Int(ReflectionUtil.Get(creature, "CurrentHp")),
            ["max_hp"] = ReflectionUtil.Int(ReflectionUtil.Get(creature, "MaxHp")),
            ["block"] = ReflectionUtil.Int(ReflectionUtil.Get(creature, "Block")),
            ["gold"] = ReflectionUtil.Int(ReflectionUtil.Get(player, "Gold")),
            ["deck"] = deck,
            ["relics"] = RelicList(ReflectionUtil.Get(player, "Relics"), "relic"),
            ["potions"] = ModelList(ReflectionUtil.Get(player, "Potions") ?? ReflectionUtil.Get(player, "PotionSlots"), "potion")
        };
        var source = new Dictionary<string, object?>();
        ContentProvenance.AddEntitySource(source, character);
        result["character_source_kind"] = source["source_kind"];
        result["character_source_assembly"] = source["source_assembly"];
        result["character_source_mod_id"] = source["source_mod_id"];
        result["character_source"] = source;
        return result;
    }

    private static Dictionary<string, object?> CombatSummary(object? player)
    {
        var pcs = ReflectionUtil.Get(player, "PlayerCombatState");
        var combatManagerType = AccessTools.TypeByName("MegaCrit.Sts2.Core.Combat.CombatManager");
        var manager = ReflectionUtil.Get(combatManagerType, "Instance");
        var state = ReflectionUtil.Call(manager, "DebugOnlyGetState");
        var aliveEnemies = ReflectionUtil.Items(ReflectionUtil.Get(state, "Enemies"))
            .Where(x => ReflectionUtil.Bool(ReflectionUtil.Get(x, "IsAlive"), true)).ToList();
        var energy = ReflectionUtil.Int(ReflectionUtil.Get(pcs, "Energy"));
        var enemies = Creatures(aliveEnemies, state);
        var result = new Dictionary<string, object?>
        {
            ["round"] = ReflectionUtil.Int(ReflectionUtil.Get(state, "RoundNumber")),
            ["turn"] = ReflectionUtil.Int(ReflectionUtil.Get(pcs, "TurnNumber")),
            ["turn_phase"] = ReflectionUtil.Get(pcs, "Phase")?.ToString(),
            ["energy"] = energy,
            ["max_energy"] = ReflectionUtil.Int(ReflectionUtil.Get(pcs, "MaxEnergy")),
            ["stars"] = ReflectionUtil.Int(ReflectionUtil.Get(pcs, "Stars")),
            ["hand"] = CardList(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "Hand"), "Cards"), "hand", true, aliveEnemies, energy),
            ["player_powers"] = ModelList(ReflectionUtil.Get(ReflectionUtil.Get(player, "Creature"), "Powers"), "power"),
            ["draw_pile_count"] = ReflectionUtil.Items(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "DrawPile"), "Cards")).Count(),
            ["discard_pile_count"] = ReflectionUtil.Items(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "DiscardPile"), "Cards")).Count(),
            ["exhaust_pile_count"] = ReflectionUtil.Items(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "ExhaustPile"), "Cards")).Count(),
            // Pile order is intentionally discarded: players may inspect contents but not draw order.
            ["draw_pile"] = VisiblePile(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "DrawPile"), "Cards"), "draw"),
            ["discard_pile"] = VisiblePile(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "DiscardPile"), "Cards"), "discard"),
            ["exhaust_pile"] = VisiblePile(ReflectionUtil.Get(ReflectionUtil.Get(pcs, "ExhaustPile"), "Cards"), "exhaust"),
            ["enemies"] = enemies,
            ["intent_capture_complete"] = enemies.All(enemy => IntentDamageComplete(enemy))
        };
        AddCharacterCombatState(result, player, pcs);
        return result;
    }

    private static List<Dictionary<string, object?>> CardList(object? cards, string prefix,
        bool combatPreview = false, List<object>? targets = null, int energy = 0)
    {
        var list = new List<Dictionary<string, object?>>();
        var i = 0;
        foreach (var card in ReflectionUtil.Items(cards))
        {
            var identity = CardIdentity.Resolve(card, string.Equals(prefix, "deck", StringComparison.Ordinal));
            var energyCost = ReflectionUtil.Get(card, "EnergyCost");
            var stats = CardStats(card, combatPreview ? "Normal" : null, ReflectionUtil.Get(card, "CurrentTarget"));
            var enchantment = NativeModelState.Enchantment(ReflectionUtil.Get(card, "Enchantment"));
            var affliction = NativeModelState.Affliction(ReflectionUtil.Get(card, "Affliction"));
            var playability = Playability(card);
            var persistent = NativeModelState.CapturePersistentState(card, false);
            var row = new Dictionary<string, object?>
            {
                ["index"] = i,
                ["instance_id"] = $"{prefix}:{i}:{RuntimeHelpers.GetHashCode(card):x8}",
                ["lineage_id"] = identity.LineageId,
                ["lineage_quality"] = identity.Quality,
                ["engine_object_ref"] = RuntimeHelpers.GetHashCode(card).ToString("x8"),
                ["id"] = ReflectionUtil.Id(card),
                ["display_name"] = ReflectionUtil.Get(card, "Title")?.ToString(),
                ["type"] = ReflectionUtil.Get(card, "Type")?.ToString(),
                ["rarity"] = ReflectionUtil.Get(card, "Rarity")?.ToString(),
                ["upgrade_level"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "CurrentUpgradeLevel")),
                ["max_upgrade_level"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "MaxUpgradeLevel")),
                ["floor_added"] = ReflectionUtil.Get(card, "FloorAddedToDeck"),
                ["cost"] = ReflectionUtil.Int(ReflectionUtil.Call(energyCost, "GetResolved"), ReflectionUtil.Int(ReflectionUtil.Get(card, "CanonicalEnergyCost"))),
                ["star_cost"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "CurrentStarCost", "BaseStarCost")),
                ["energy_cost"] = EnergyCost(card, energyCost),
                ["star_cost_state"] = StarCost(card),
                ["target_type"] = ReflectionUtil.Get(card, "TargetType")?.ToString(),
                ["can_play"] = playability["can_play"],
                ["playability"] = playability,
                ["keywords"] = ReflectionUtil.Items(ReflectionUtil.Get(card, "Keywords"))
                    .Select(x => x.ToString()).Where(x => !string.Equals(x, "None", StringComparison.OrdinalIgnoreCase)).ToList(),
                ["tags"] = ReflectionUtil.Items(ReflectionUtil.Get(card, "Tags"))
                    .Select(x => x.ToString()).Where(x => !string.Equals(x, "None", StringComparison.OrdinalIgnoreCase)).ToList(),
                ["stats"] = stats.Count > 0 ? stats : null,
                ["dynamic_vars"] = new Dictionary<string, object?>
                {
                    ["effective"] = DynamicValues(card, false),
                    ["preview"] = stats
                },
                ["runtime_flags"] = new Dictionary<string, object?>
                {
                    ["exhaust_on_next_play"] = ReflectionUtil.Bool(ReflectionUtil.Get(card, "ExhaustOnNextPlay")),
                    ["retain_this_turn"] = ReflectionUtil.Bool(ReflectionUtil.Get(card, "ShouldRetainThisTurn")),
                    ["sly_this_turn"] = ReflectionUtil.Bool(ReflectionUtil.Get(card, "IsSlyThisTurn"))
                },
                ["enchantment"] = enchantment.Count == 0 ? null : enchantment,
                ["affliction"] = affliction.Count == 0 ? null : affliction,
                ["persistent_state"] = persistent.State,
                ["persistent_state_capture_quality"] = persistent.Quality,
                ["persistent_state_capture_error"] = persistent.Error,
                ["runtime_state"] = NativeModelState.RuntimeState(card, ContentStateRegistry.CardRuntime(card)),
                ["state_schema"] = NativeModelState.SchemaVersion
            };
            if (combatPreview && targets is not null)
            {
                var damageByTarget = DamageByTarget(card, targets, energy);
                if (damageByTarget.Count > 0) row["damage_by_target"] = damageByTarget;
            }
            ContentProvenance.AddEntitySource(row, card);
            list.Add(row);
            i++;
        }
        return list;
    }

    private static List<Dictionary<string, object?>> RelicList(object? relics, string prefix)
    {
        var list = new List<Dictionary<string, object?>>();
        var index = 0;
        foreach (var relic in ReflectionUtil.Items(relics))
        {
            var row = NativeModelState.Relic(relic);
            row["index"] = index;
            row["instance_id"] = $"{prefix}:{index}:{RuntimeHelpers.GetHashCode(relic):x8}";
            list.Add(row);
            index++;
        }
        return list;
    }

    private static List<Dictionary<string, object?>> ModelList(object? models, string prefix)
    {
        var list = new List<Dictionary<string, object?>>();
        var i = 0;
        foreach (var model in ReflectionUtil.Items(models))
        {
            var row = new Dictionary<string, object?>
            {
                ["index"] = i,
                ["instance_id"] = $"{prefix}:{i}:{RuntimeHelpers.GetHashCode(model):x8}",
                ["id"] = ReflectionUtil.Id(model),
                ["amount"] = ReflectionUtil.Get(model, "Amount", "StackCount", "DisplayAmount"),
                ["target_type"] = ReflectionUtil.Get(model, "TargetType")?.ToString(),
                ["vars"] = DynamicValues(model, false)
            };
            ContentProvenance.AddEntitySource(row, model);
            list.Add(row);
            i++;
        }
        return list;
    }

    private static List<Dictionary<string, object?>> AuditCards(object? cards) =>
        ReflectionUtil.Items(cards).Select(NativeModelState.AuditCard).ToList();

    private static Dictionary<string, object?> EnergyCost(object card, object? energyCost)
    {
        var modifiers = ReflectionUtil.Items(ReflectionUtil.Get(energyCost, "_localModifiers")).Select(modifier =>
            new Dictionary<string, object?>
            {
                ["amount"] = ReflectionUtil.Int(ReflectionUtil.Get(modifier, "Amount")),
                ["kind"] = ReflectionUtil.Get(modifier, "Type")?.ToString(),
                ["expires"] = ReflectionUtil.Get(modifier, "Expiration")?.ToString(),
                ["reduce_only"] = ReflectionUtil.Bool(ReflectionUtil.Get(modifier, "IsReduceOnly"))
            }).ToList();
        return new Dictionary<string, object?>
        {
            ["canonical"] = ReflectionUtil.Int(ReflectionUtil.Get(energyCost, "Canonical"), ReflectionUtil.Int(ReflectionUtil.Get(card, "CanonicalEnergyCost"))),
            ["current"] = ReflectionUtil.Int(ReflectionUtil.Call(energyCost, "GetResolved"), ReflectionUtil.Int(ReflectionUtil.Get(card, "CanonicalEnergyCost"))),
            ["costs_x"] = ReflectionUtil.Bool(ReflectionUtil.Get(energyCost, "CostsX")),
            ["captured_x"] = ReflectionUtil.Get(energyCost, "CapturedXValue"),
            ["modifiers"] = modifiers
        };
    }

    private static Dictionary<string, object?> StarCost(object card)
    {
        var modifiers = ReflectionUtil.Items(ReflectionUtil.Get(card, "_temporaryStarCosts")).Select(modifier =>
            new Dictionary<string, object?>
            {
                ["cost"] = ReflectionUtil.Int(ReflectionUtil.Get(modifier, "Cost")),
                ["clears_when_turn_ends"] = ReflectionUtil.Bool(ReflectionUtil.Get(modifier, "ClearsWhenTurnEnds")),
                ["clears_when_card_is_played"] = ReflectionUtil.Bool(ReflectionUtil.Get(modifier, "ClearsWhenCardIsPlayed"))
            }).ToList();
        return new Dictionary<string, object?>
        {
            ["canonical"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "CanonicalStarCost")),
            ["base"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "BaseStarCost")),
            ["current"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "CurrentStarCost")),
            ["costs_x"] = ReflectionUtil.Bool(ReflectionUtil.Get(card, "HasStarCostX")),
            ["last_spent"] = ReflectionUtil.Int(ReflectionUtil.Get(card, "LastStarsSpent")),
            ["modifiers"] = modifiers
        };
    }

    private static Dictionary<string, object?> Playability(object card)
    {
        var result = new Dictionary<string, object?>
        {
            ["can_play"] = ReflectionUtil.Bool(ReflectionUtil.Call(card, "CanPlay")),
            ["reason"] = null,
            ["reason_source_id"] = null
        };
        try
        {
            var method = card.GetType().GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(candidate => candidate.Name == "CanPlay" && candidate.GetParameters().Length == 2
                    && candidate.GetParameters().All(parameter => parameter.ParameterType.IsByRef));
            if (method is null) return result;
            var args = new object?[] { null, null };
            result["can_play"] = ReflectionUtil.Bool(method.Invoke(card, args));
            result["reason"] = args[0]?.ToString();
            result["reason_source_id"] = ReflectionUtil.Id(args[1]);
        }
        catch { }
        return result;
    }

    private static List<Dictionary<string, object?>> Creatures(object? creatures, object? combatState)
    {
        var rows = new List<Dictionary<string, object?>>();
        var i = 0;
        foreach (var creature in ReflectionUtil.Items(creatures))
        {
            if (!ReflectionUtil.Bool(ReflectionUtil.Get(creature, "IsAlive"), true)) continue;
            var monster = ReflectionUtil.Get(creature, "Monster");
            var row = new Dictionary<string, object?>
            {
                ["index"] = i++, ["id"] = ReflectionUtil.Id(monster) ?? ReflectionUtil.Id(creature),
                ["combat_id"] = ReflectionUtil.Get(creature, "CombatId")?.ToString(),
                ["hp"] = ReflectionUtil.Int(ReflectionUtil.Get(creature, "CurrentHp")),
                ["max_hp"] = ReflectionUtil.Int(ReflectionUtil.Get(creature, "MaxHp")),
                ["block"] = ReflectionUtil.Int(ReflectionUtil.Get(creature, "Block")),
                ["intends_attack"] = ReflectionUtil.Bool(ReflectionUtil.Get(monster, "IntendsToAttack")),
                ["intent"] = Intent(monster, creature, combatState),
                ["powers"] = ModelList(ReflectionUtil.Get(creature, "Powers"), "power")
            };
            ContentProvenance.AddEntitySource(row, monster ?? creature);
            rows.Add(row);
        }
        return rows;
    }

    private static List<Dictionary<string, object?>> Intent(object? monster, object creature, object? combatState)
    {
        var move = ReflectionUtil.Get(monster, "NextMove");
        var playerCreatures = ReflectionUtil.Get(combatState, "PlayerCreatures");
        return ReflectionUtil.Items(ReflectionUtil.Get(move, "Intents")).Select(x =>
        {
            var type = ReflectionUtil.Get(x, "IntentType")?.ToString();
            var hits = Math.Max(1, ReflectionUtil.Int(ReflectionUtil.Get(x, "Repeats"), 1));
            var row = new Dictionary<string, object?> { ["type"] = type, ["repeats"] = hits, ["hits"] = hits };
            var label = LocStringSummary(ReflectionUtil.Call(x, "GetIntentLabel", playerCreatures, creature));
            var description = LocStringSummary(ReflectionUtil.Call(x, "GetIntentDescription", playerCreatures, creature));
            if (label.Count > 0) row["label"] = label;
            if (description.Count > 0) row["description"] = description;
            var single = ReflectionUtil.Call(x, "GetSingleDamage", playerCreatures, creature);
            var total = ReflectionUtil.Call(x, "GetTotalDamage", playerCreatures, creature);
            var isAttack = single is not null || total is not null
                || string.Equals(type, "Attack", StringComparison.OrdinalIgnoreCase)
                || string.Equals(type, "DeathBlow", StringComparison.OrdinalIgnoreCase);
            row["is_attack"] = isAttack;
            if (isAttack)
            {
                if (hits > 1)
                {
                    if (single is not null) row["damage"] = ReflectionUtil.Int(single);
                    if (total is not null) row["total_damage"] = ReflectionUtil.Int(total);
                    else if (single is not null) row["total_damage"] = ReflectionUtil.Int(single) * hits;
                }
                else
                {
                    var damage = total ?? single;
                    if (damage is not null)
                    {
                        row["damage"] = ReflectionUtil.Int(damage);
                        row["total_damage"] = ReflectionUtil.Int(damage);
                    }
                }
            }
            return row;
        }).ToList();
    }

    private static bool IntentDamageComplete(Dictionary<string, object?> enemy)
    {
        if (!ReflectionUtil.Bool(enemy.GetValueOrDefault("intends_attack"))) return true;
        var attacks = ((enemy.GetValueOrDefault("intent") as List<Dictionary<string, object?>>) ?? new())
            .Where(intent => ReflectionUtil.Bool(intent.GetValueOrDefault("is_attack"))
                || string.Equals(intent.GetValueOrDefault("type")?.ToString(), "Attack", StringComparison.OrdinalIgnoreCase)
                || string.Equals(intent.GetValueOrDefault("type")?.ToString(), "DeathBlow", StringComparison.OrdinalIgnoreCase)).ToList();
        return attacks.Count > 0 && attacks.All(intent => intent.GetValueOrDefault("damage") is not null);
    }

    private static Dictionary<string, object?> LocStringSummary(object? locString)
    {
        if (locString is null) return new();
        var result = new Dictionary<string, object?>();
        var table = ReflectionUtil.Get(locString, "LocTable")?.ToString();
        var key = ReflectionUtil.Get(locString, "LocEntryKey")?.ToString();
        if (!string.IsNullOrWhiteSpace(table)) result["table"] = table;
        if (!string.IsNullOrWhiteSpace(key)) result["key"] = key;
        var variables = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var pair in ReflectionUtil.Items(ReflectionUtil.Get(locString, "Variables")))
        {
            var name = ReflectionUtil.Get(pair, "Key")?.ToString();
            if (!string.IsNullOrWhiteSpace(name)) variables[name] = VisibleValue(ReflectionUtil.Get(pair, "Value"));
        }
        if (variables.Count > 0) result["variables"] = variables;
        var formatted = ReflectionUtil.Call(locString, "GetFormattedText")?.ToString();
        if (!string.IsNullOrWhiteSpace(formatted)) result["display_text"] = formatted;
        return result;
    }

    private static object? VisibleValue(object? value)
    {
        if (value is null || value is string || value is bool || value is byte || value is sbyte
            || value is short || value is ushort || value is int || value is uint || value is long
            || value is ulong || value is float || value is double || value is decimal) return value;
        if (value.GetType().IsEnum) return value.ToString();
        return ReflectionUtil.Id(value);
    }

    private static List<Dictionary<string, object?>> VisiblePile(object? cards, string prefix)
    {
        var rows = CardList(cards, prefix);
        return rows.GroupBy(row => string.Join("|", new[]
            {
                row.GetValueOrDefault("id")?.ToString(), row.GetValueOrDefault("upgrade_level")?.ToString(),
                row.GetValueOrDefault("cost")?.ToString(), row.GetValueOrDefault("star_cost")?.ToString(),
                NestedId(row, "enchantment"), NestedId(row, "affliction")
            }))
            .OrderBy(group => group.Key, StringComparer.Ordinal)
            .Select(group =>
            {
                var row = new Dictionary<string, object?>(group.First());
                row.Remove("index"); row.Remove("instance_id"); row.Remove("can_play");
                row["count"] = group.Count();
                return row;
            }).ToList();
    }

    private static string? NestedId(Dictionary<string, object?> row, string key) =>
        (row.GetValueOrDefault(key) as Dictionary<string, object?>)?.GetValueOrDefault("id")?.ToString();

    private static void ApplyNativeStateQuality(Dictionary<string, object?> root)
    {
        var errors = new List<string>();
        CollectNativeStateErrors(root, errors);
        if (errors.Count == 0) return;
        root["capture_quality"] = "partial";
        var existing = ReflectionUtil.Items(root.GetValueOrDefault("capture_errors"))
            .Select(value => value.ToString() ?? "unknown").ToList();
        existing.AddRange(errors.Distinct(StringComparer.Ordinal));
        root["capture_errors"] = existing;
    }

    private static void CollectNativeStateErrors(object? value, List<string> errors)
    {
        if (value is null || value is string) return;
        if (value is IDictionary dictionary)
        {
            if (dictionary.Contains("persistent_state_capture_quality")
                && string.Equals(dictionary["persistent_state_capture_quality"]?.ToString(), "partial", StringComparison.Ordinal))
                errors.Add("native_state: " + (dictionary["persistent_state_capture_error"]?.ToString() ?? "unknown"));
            foreach (DictionaryEntry entry in dictionary) CollectNativeStateErrors(entry.Value, errors);
            return;
        }
        if (value is IEnumerable enumerable)
            foreach (var item in enumerable) CollectNativeStateErrors(item, errors);
    }

    private static Dictionary<string, object?> CardStats(object card, string? previewMode, object? target)
    {
        var vars = ReflectionUtil.Get(card, "DynamicVars");
        if (vars is null) return new();
        try
        {
            ReflectionUtil.Call(vars, "ClearPreview");
            if (previewMode is not null)
            {
                var modeType = AccessTools.TypeByName("MegaCrit.Sts2.Core.Entities.Cards.CardPreviewMode");
                var mode = modeType is null ? null : Enum.Parse(modeType, previewMode);
                if (mode is not null) ReflectionUtil.Call(card, "UpdateDynamicVarPreview", mode, target, vars);
            }
            return DynamicValues(vars, previewMode is not null);
        }
        finally { ReflectionUtil.Call(vars, "ClearPreview"); }
    }

    private static Dictionary<string, object?> DynamicValues(object ownerOrVars, bool preview)
    {
        var vars = ReflectionUtil.Get(ownerOrVars, "DynamicVars") ?? ownerOrVars;
        var result = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        foreach (var value in ReflectionUtil.Items(ReflectionUtil.Get(vars, "Values")))
        {
            var name = ReflectionUtil.Get(value, "Name")?.ToString()?.ToLowerInvariant();
            if (string.IsNullOrWhiteSpace(name)) continue;
            result[name] = ReflectionUtil.Int(ReflectionUtil.Get(value, preview ? "PreviewValue" : "BaseValue", "BaseValue"));
        }
        return result;
    }

    private static List<Dictionary<string, object?>> DamageByTarget(object card, List<object> targets, int energy)
    {
        var type = ReflectionUtil.Get(card, "Type")?.ToString();
        var targetType = ReflectionUtil.Get(card, "TargetType")?.ToString();
        if (!string.Equals(type, "Attack", StringComparison.OrdinalIgnoreCase)
            || targetType is null || (!targetType.Contains("AnyEnemy", StringComparison.OrdinalIgnoreCase)
                && !targetType.Contains("AllEnemies", StringComparison.OrdinalIgnoreCase))) return new();
        var rows = new List<Dictionary<string, object?>>();
        for (var i = 0; i < targets.Count; i++)
        {
            var target = targets[i];
            var stats = CardStats(card, "MultiCreatureTargeting", target);
            var damage = stats.TryGetValue("calculateddamage", out var calculated) && ReflectionUtil.Int(calculated) > 0
                ? ReflectionUtil.Int(calculated) : ReflectionUtil.Int(stats.GetValueOrDefault("damage"), -1);
            var hits = Math.Max(1, ReflectionUtil.Int(stats.GetValueOrDefault("repeat"), 1));
            if (hits == 1 && ReflectionUtil.Bool(ReflectionUtil.Get(ReflectionUtil.Get(card, "EnergyCost"), "CostsX")))
                hits = Math.Max(0, energy);
            var monster = ReflectionUtil.Get(target, "Monster");
            var row = new Dictionary<string, object?>
            {
                ["target_index"] = i, ["target_id"] = ReflectionUtil.Id(monster) ?? ReflectionUtil.Id(target),
                ["target_combat_id"] = ReflectionUtil.Get(target, "CombatId")?.ToString(), ["hits"] = hits
            };
            if (damage >= 0) { row["damage"] = damage; row["total_damage"] = damage * hits; }
            rows.Add(row);
        }
        return rows;
    }

    private static void AddCharacterCombatState(Dictionary<string, object?> result, object? player, object? pcs)
    {
        var orbQueue = ReflectionUtil.Get(pcs, "OrbQueue");
        var orbs = ReflectionUtil.Items(ReflectionUtil.Get(orbQueue, "Orbs")).Select((orb, i) => new Dictionary<string, object?>
        {
            ["index"] = i, ["id"] = ReflectionUtil.Id(orb),
            ["passive"] = ReflectionUtil.Int(ReflectionUtil.Get(orb, "PassiveVal")),
            ["evoke"] = ReflectionUtil.Int(ReflectionUtil.Get(orb, "EvokeVal"))
        }).ToList();
        if (orbs.Count > 0) { result["orbs"] = orbs; result["orb_slots"] = ReflectionUtil.Int(ReflectionUtil.Get(orbQueue, "Capacity")); }
        var osty = ReflectionUtil.Get(player, "Osty");
        if (osty is not null) result["osty"] = new Dictionary<string, object?>
        {
            ["id"] = ReflectionUtil.Id(ReflectionUtil.Get(osty, "Monster")),
            ["hp"] = ReflectionUtil.Int(ReflectionUtil.Get(osty, "CurrentHp")),
            ["max_hp"] = ReflectionUtil.Int(ReflectionUtil.Get(osty, "MaxHp")),
            ["block"] = ReflectionUtil.Int(ReflectionUtil.Get(osty, "Block")),
            ["alive"] = ReflectionUtil.Bool(ReflectionUtil.Get(osty, "IsAlive"))
        };
    }

    private static List<Dictionary<string, object?>> RewardRows(object? value)
    {
        var rows = new List<Dictionary<string, object?>>();
        foreach (var reward in ReflectionUtil.Items(value)) AddRewardRows(rows, reward);
        return rows.OrderBy(row => ReflectionUtil.Int(row.GetValueOrDefault("index"), int.MaxValue))
            .ThenBy(row => row.GetValueOrDefault("type")?.ToString(), StringComparer.Ordinal).ToList();
    }

    private static void AddRewardRows(List<Dictionary<string, object?>> rows, object reward)
    {
        var children = ReflectionUtil.Items(ReflectionUtil.Get(reward, "Rewards")).ToList();
        if (children.Count > 0)
        {
            foreach (var child in children) AddRewardRows(rows, child);
            return;
        }
        var entity = ReflectionUtil.Get(reward, "Potion", "Relic", "Card", "_card", "ClaimedPotion", "ClaimedRelic");
        var cards = CardList(ReflectionUtil.Get(reward, "Cards"), "reward_card");
        var type = ReflectionUtil.Get(reward, "RewardType")?.ToString() ?? reward.GetType().Name;
        var row = new Dictionary<string, object?>
        {
            ["index"] = ReflectionUtil.Int(ReflectionUtil.Get(reward, "RewardsSetIndex"), rows.Count),
            ["type"] = type,
            ["id"] = ReflectionUtil.Id(entity) ?? type,
            ["amount"] = ReflectionUtil.Get(reward, "Amount"),
            ["selected"] = ReflectionUtil.Bool(ReflectionUtil.Get(reward, "SuccessfullySelected")),
            ["can_skip"] = ReflectionUtil.Bool(ReflectionUtil.Get(reward, "CanSkip"), true),
            ["cards"] = cards.Count > 0 ? cards : null
        };
        ContentProvenance.AddEntitySource(row, entity ?? reward);
        rows.Add(row);
    }

    private static object? FindAncestorWithMember(object start, string member)
    {
        object? cursor = start;
        for (var depth = 0; depth < 16 && cursor is not null; depth++)
        {
            if (ReflectionUtil.Get(cursor, member) is not null) return cursor;
            cursor = ReflectionUtil.Call(cursor, "GetParent");
        }
        return null;
    }

    private static bool IsCombatInProgress()
    {
        var manager = ReflectionUtil.Get(AccessTools.TypeByName("MegaCrit.Sts2.Core.Combat.CombatManager"), "Instance");
        return ReflectionUtil.Bool(ReflectionUtil.Get(manager, "IsInProgress"));
    }

    private static Dictionary<string, object?> MapSummary(object state)
    {
        var map = ReflectionUtil.Get(state, "Map");
        var nodes = new List<Dictionary<string, object?>>();
        var all = ReflectionUtil.Call(map, "GetAllMapPoints") ?? ReflectionUtil.Get(map, "Grid");
        foreach (var point in ReflectionUtil.Items(all))
        {
            var coord = ReflectionUtil.Get(point, "coord", "Coord");
            nodes.Add(new Dictionary<string, object?>
            {
                ["coord"] = Coord(coord), ["type"] = ReflectionUtil.Get(point, "PointType")?.ToString(),
                ["children"] = ReflectionUtil.Items(ReflectionUtil.Get(point, "Children"))
                    .Select(x => Coord(ReflectionUtil.Get(x, "coord", "Coord"))).ToList()
            });
        }
        return new Dictionary<string, object?>
        {
            ["current"] = Coord(ReflectionUtil.Get(state, "CurrentMapCoord")),
            ["visited"] = ReflectionUtil.Items(ReflectionUtil.Get(state, "VisitedMapCoords")).Select(Coord).ToList(),
            ["nodes"] = nodes
        };
    }

    private static object? EventSummary(object state)
    {
        var room = ReflectionUtil.Get(state, "CurrentRoom");
        var ev = ReflectionUtil.Get(room, "LocalMutableEvent", "CanonicalEvent");
        var options = ReflectionUtil.Items(ReflectionUtil.Get(ev, "CurrentOptions")).Select((x, i) => new Dictionary<string, object?>
        {
            ["index"] = i, ["id"] = ReflectionUtil.Id(x), ["locked"] = ReflectionUtil.Bool(ReflectionUtil.Get(x, "IsLocked")),
            ["proceed"] = ReflectionUtil.Bool(ReflectionUtil.Get(x, "IsProceed")), ["title"] = ReflectionUtil.Get(x, "Title")?.ToString()
        }).ToList();
        return new Dictionary<string, object?> { ["id"] = ReflectionUtil.Id(ev), ["options"] = options };
    }

    private static object? RestSummary(object state)
    {
        var room = ReflectionUtil.Get(state, "CurrentRoom");
        return ReflectionUtil.Items(ReflectionUtil.Get(room, "Options")).Select((x, i) => new Dictionary<string, object?>
        {
            ["index"] = i, ["id"] = ReflectionUtil.Id(x), ["enabled"] = ReflectionUtil.Bool(ReflectionUtil.Get(x, "IsEnabled"), true)
        }).ToList();
    }

    private static object? ShopSummary(object state, object? player)
    {
        var room = ReflectionUtil.Get(state, "CurrentRoom");
        var inventory = ReflectionUtil.Items(ReflectionUtil.Get(room, "Inventories"))
            .FirstOrDefault(x => ReferenceEquals(ReflectionUtil.Get(x, "Player"), player))
            ?? ReflectionUtil.Items(ReflectionUtil.Get(room, "Inventories")).FirstOrDefault();
        return ShopRows(inventory);
    }

    private static List<Dictionary<string, object?>> ShopRows(object? inventory)
    {
        return ReflectionUtil.Items(ReflectionUtil.Get(inventory, "AllEntries")).Select((x, i) => new Dictionary<string, object?>
        {
            ["index"] = i, ["kind"] = x.GetType().Name, ["id"] = EntryId(x),
            ["cost"] = ReflectionUtil.Int(ReflectionUtil.Get(x, "Cost")), ["stocked"] = ReflectionUtil.Bool(ReflectionUtil.Get(x, "IsStocked"), true),
            ["affordable"] = ReflectionUtil.Bool(ReflectionUtil.Get(x, "EnoughGold"))
        }).ToList();
    }

    private static string? EntryId(object entry)
    {
        var model = ReflectionUtil.Get(entry, "Model");
        var creation = ReflectionUtil.Get(entry, "CreationResult");
        return ReflectionUtil.Id(model) ?? ReflectionUtil.Id(ReflectionUtil.Get(creation, "Card", "CreatedCard")) ?? entry.GetType().Name;
    }

    private static List<Dictionary<string, object?>> LegalActions(string phase, Dictionary<string, object?> obs, object state, object? player)
    {
        var actions = new List<Dictionary<string, object?>>();
        if (phase == "map_select")
        {
            var map = (Dictionary<string, object?>)obs["visible_map"]!;
            var current = ReflectionUtil.Get(state, "CurrentMapPoint");
            var choices = current is null ? new[] { ReflectionUtil.Get(ReflectionUtil.Get(state, "Map"), "StartingMapPoint")! }
                : ReflectionUtil.Items(ReflectionUtil.Get(current, "Children"));
            foreach (var choice in choices.Where(x => x is not null))
                actions.Add(Action("select_map_node", new Dictionary<string, object?> { ["coord"] = Coord(ReflectionUtil.Get(choice, "coord", "Coord")) }));
        }
        else if (phase == "combat_play")
        {
            var combat = (Dictionary<string, object?>)obs["combat"]!;
            var hand = (List<Dictionary<string, object?>>)combat["hand"]!;
            var enemies = (List<Dictionary<string, object?>>)combat["enemies"]!;
            foreach (var card in hand.Where(x => ReflectionUtil.Bool(x["can_play"])))
            {
                var target = card["target_type"]?.ToString();
                if (target is not null && target.Contains("AnyEnemy", StringComparison.OrdinalIgnoreCase))
                    foreach (var enemy in enemies)
                        actions.Add(Action("play_card", new() { ["card_instance_id"] = card["instance_id"], ["target_index"] = enemy["index"] }));
                else actions.Add(Action("play_card", new() { ["card_instance_id"] = card["instance_id"] }));
            }
            actions.Add(Action("end_turn", new()));
            foreach (var potion in ((List<Dictionary<string, object?>>)((Dictionary<string, object?>)obs["player"]!)["potions"]!).Where(x => x["id"] is not null))
            {
                var potionTarget = potion.GetValueOrDefault("target_type")?.ToString();
                if (potionTarget?.Contains("AnyEnemy", StringComparison.OrdinalIgnoreCase) == true)
                    foreach (var enemy in enemies)
                        actions.Add(Action("use_potion", new()
                        {
                            ["potion_instance_id"] = potion["instance_id"], ["target_index"] = enemy["index"],
                            ["target_id"] = enemy["id"], ["target_combat_id"] = enemy["combat_id"]
                        }));
                else actions.Add(Action("use_potion", new() { ["potion_instance_id"] = potion["instance_id"] }));
                actions.Add(Action("discard_potion", new()
                {
                    ["potion_instance_id"] = potion["instance_id"], ["potion_id"] = potion["id"]
                }));
            }
        }
        else if (phase == "potion_manage")
        {
            foreach (var potion in ((List<Dictionary<string, object?>>)((Dictionary<string, object?>)obs["player"]!)["potions"]!).Where(x => x["id"] is not null))
                actions.Add(Action("discard_potion", new()
                {
                    ["potion_instance_id"] = potion["instance_id"], ["potion_id"] = potion["id"]
                }));
        }
        else if (phase == "event_choice")
        {
            var ev = (Dictionary<string, object?>)obs["event"]!;
            foreach (var option in ((List<Dictionary<string, object?>>)ev["options"]!).Where(x => !ReflectionUtil.Bool(x["locked"])))
                actions.Add(Action("choose_event_option", new() { ["index"] = option["index"], ["id"] = option["id"] }));
        }
        else if (phase == "rest_site")
        {
            foreach (var option in ((List<Dictionary<string, object?>>)obs["rest_site"]!).Where(x => ReflectionUtil.Bool(x["enabled"])))
                actions.Add(Action("choose_rest_option", new() { ["index"] = option["index"], ["id"] = option["id"] }));
        }
        else if (phase == "shop")
        {
            foreach (var entry in ((List<Dictionary<string, object?>>)obs["shop"]!).Where(x => ReflectionUtil.Bool(x["stocked"]) && ReflectionUtil.Bool(x["affordable"])))
            {
                var shopAction = entry["kind"]?.ToString()?.Contains("CardRemoval", StringComparison.OrdinalIgnoreCase) == true
                    ? "remove_card" : "buy_shop_item";
                actions.Add(Action(shopAction, new() { ["index"] = entry["index"], ["id"] = entry["id"], ["cost"] = entry["cost"] }));
            }
            actions.Add(Action("leave_shop", new()));
        }
        return actions;
    }

    private static Dictionary<string, object?> Action(string id, Dictionary<string, object?> args) => new() { ["action_id"] = id, ["args"] = args };

    private static object? Coord(object? coord)
    {
        if (coord is null) return null;
        var value = ReflectionUtil.Get(coord, "Value") ?? coord;
        return new Dictionary<string, object?> { ["col"] = ReflectionUtil.Int(ReflectionUtil.Get(value, "col", "Col")), ["row"] = ReflectionUtil.Int(ReflectionUtil.Get(value, "row", "Row")) };
    }
}
