using System.Runtime.CompilerServices;

namespace Sts2HumanRecorder;

internal static class ActionEncoder
{
    public static Dictionary<string, object?> Encode(string actionId, object? instance, object?[] args)
    {
        var payload = new Dictionary<string, object?> { ["action_id"] = actionId, ["args"] = new Dictionary<string, object?>() };
        var values = (Dictionary<string, object?>)payload["args"]!;
        switch (actionId)
        {
            case "select_map_node": values["coord"] = Coord(args.FirstOrDefault()); break;
            case "play_card":
                var action = args.FirstOrDefault();
                var card = args.Skip(2).FirstOrDefault() ?? ReflectionUtil.Get(action, "_card", "Card");
                values["card_id"] = ReflectionUtil.Id(card) ?? ReflectionUtil.Get(action, "CardModelId")?.ToString();
                values["card_object_id"] = card is null ? null : RuntimeHelpers.GetHashCode(card).ToString("x8");
                var target = ReflectionUtil.Get(action, "Target");
                values["target_combat_id"] = ReflectionUtil.Get(target, "CombatId")?.ToString();
                values["target_id"] = ReflectionUtil.Id(ReflectionUtil.Get(target, "Monster")) ?? ReflectionUtil.Id(target);
                AddSource(values, card);
                break;
            case "choose_event_option":
            case "choose_rest_option":
            case "choose_reward_alternative": values["index"] = ReflectionUtil.Int(args.FirstOrDefault()); break;
            case "choose_card_reward":
            case "choose_card":
                var holder = args.FirstOrDefault();
                var chosen = ReflectionUtil.Get(holder, "CardModel", "Card", "Model");
                values["card_id"] = ReflectionUtil.Id(chosen);
                if (chosen is not null)
                {
                    values["card_object_id"] = RuntimeHelpers.GetHashCode(chosen).ToString("x8");
                    values["card_lineage_id"] = CardIdentity.Resolve(chosen, false).LineageId;
                }
                AddSource(values, chosen);
                break;
            case "confirm_card_selection":
                var selectedCards = ReflectionUtil.Items(ReflectionUtil.Get(instance, "_selectedCards")).ToList();
                values["selected_card_ids"] = selectedCards.Select(ReflectionUtil.Id).Where(x => x is not null).ToList();
                values["selected_card_instance_ids"] = selectedCards.Select(card =>
                    $"selected:{RuntimeHelpers.GetHashCode(card):x8}").ToList();
                values["selected_card_lineage_ids"] = selectedCards.Select(card =>
                    CardIdentity.Resolve(card, false).LineageId).ToList();
                values["selected_card_sources"] = selectedCards.Select(Source).ToList();
                break;
            case "select_bundle":
                var selectedBundle = ReflectionUtil.Get(instance, "_selectedBundle");
                var bundleCards = ReflectionUtil.Items(ReflectionUtil.Get(selectedBundle, "Bundle")).ToList();
                var bundles = ReflectionUtil.Items(ReflectionUtil.Get(instance, "_bundles")).ToList();
                values["bundle_index"] = bundles.FindIndex(bundle => ReferenceEquals(bundle, ReflectionUtil.Get(selectedBundle, "Bundle")));
                values["card_ids"] = bundleCards.Select(ReflectionUtil.Id).Where(x => x is not null).ToList();
                values["card_sources"] = bundleCards.Select(Source).ToList();
                break;
            case "choose_relic":
                var relicHolder = args.FirstOrDefault();
                var relicValue = ReflectionUtil.Get(relicHolder, "Model", "RelicModel", "Relic");
                var chosenRelic = ReflectionUtil.Get(relicValue, "Model", "RelicModel") ?? relicValue;
                values["relic_id"] = ReflectionUtil.Id(chosenRelic);
                values["relic_object_id"] = chosenRelic is null ? null : RuntimeHelpers.GetHashCode(chosenRelic).ToString("x8");
                AddSource(values, chosenRelic);
                break;
            case "buy_shop_item":
                values["kind"] = instance?.GetType().Name;
                values["id"] = ReflectionUtil.Id(ReflectionUtil.Get(instance, "Model"))
                    ?? ReflectionUtil.Id(ReflectionUtil.Get(ReflectionUtil.Get(instance, "CreationResult"), "Card", "CreatedCard"));
                values["cost"] = ReflectionUtil.Int(ReflectionUtil.Get(instance, "Cost"));
                break;
            case "use_potion":
                var potion = args.FirstOrDefault();
                values["potion_id"] = ReflectionUtil.Id(potion);
                var potionTarget = args.Skip(1).FirstOrDefault();
                values["potion_object_id"] = potion is null ? null : RuntimeHelpers.GetHashCode(potion).ToString("x8");
                values["target_combat_id"] = ReflectionUtil.Get(potionTarget, "CombatId")?.ToString();
                values["target_id"] = ReflectionUtil.Id(ReflectionUtil.Get(potionTarget, "Monster")) ?? ReflectionUtil.Id(potionTarget);
                AddSource(values, potion);
                break;
            case "discard_potion":
                var potionNode = ReflectionUtil.Get(instance, "Potion");
                var discardedPotion = ReflectionUtil.Get(potionNode, "Model", "PotionModel", "Potion") ?? potionNode;
                values["potion_id"] = ReflectionUtil.Id(discardedPotion);
                values["potion_object_id"] = discardedPotion is null ? null : RuntimeHelpers.GetHashCode(discardedPotion).ToString("x8");
                AddSource(values, discardedPotion);
                break;
            case "select_reward":
                var reward = ReflectionUtil.Get(instance, "Reward");
                var rewardEntity = RewardEntity(reward);
                values["reward_index"] = ReflectionUtil.Int(ReflectionUtil.Get(reward, "RewardsSetIndex"), -1);
                values["reward_type"] = ReflectionUtil.Get(reward, "RewardType")?.ToString() ?? reward?.GetType().Name;
                values["reward_id"] = ReflectionUtil.Id(rewardEntity) ?? values["reward_type"];
                AddSource(values, rewardEntity ?? reward);
                break;
            case "select_treasure_relic":
                var relicNode = ReflectionUtil.Get(instance, "Relic");
                var relic = ReflectionUtil.Get(relicNode, "Model", "RelicModel", "Relic") ?? relicNode;
                values["relic_index"] = ReflectionUtil.Int(ReflectionUtil.Get(instance, "Index"), -1);
                values["relic_id"] = ReflectionUtil.Id(relic);
                AddSource(values, relic);
                break;
            case "proceed":
            case "open_treasure":
            case "skip_treasure_relic":
            case "end_turn":
            case "leave_shop":
            case "skip":
            case "skip_card_selection": break;
            case "remove_card":
                values["cost"] = ReflectionUtil.Int(args.FirstOrDefault());
                break;
            default:
                for (var i = 0; i < args.Length; i++) values[$"arg_{i}"] = SafeArg(args[i]);
                break;
        }
        return payload;
    }

    private static object? SafeArg(object? value)
    {
        if (value is null || value is string || value.GetType().IsPrimitive || value.GetType().IsEnum) return value?.ToString();
        return new Dictionary<string, object?> { ["type"] = value.GetType().FullName, ["id"] = ReflectionUtil.Id(value) };
    }

    private static object? Coord(object? coord)
    {
        if (coord is null) return null;
        var value = ReflectionUtil.Get(coord, "Value") ?? coord;
        return new Dictionary<string, object?> { ["col"] = ReflectionUtil.Int(ReflectionUtil.Get(value, "col", "Col")), ["row"] = ReflectionUtil.Int(ReflectionUtil.Get(value, "row", "Row")) };
    }

    private static void AddSource(Dictionary<string, object?> values, object? entity)
    {
        var source = Source(entity);
        values["source_kind"] = source["source_kind"];
        values["source_assembly"] = source["source_assembly"];
        values["source_mod_id"] = source["source_mod_id"];
    }

    private static Dictionary<string, object?> Source(object? entity)
    {
        var source = new Dictionary<string, object?>();
        ContentProvenance.AddEntitySource(source, entity);
        return source;
    }

    private static object? RewardEntity(object? reward) => ReflectionUtil.Get(reward,
        "Potion", "Relic", "Card", "_card", "ClaimedPotion", "ClaimedRelic");
}
