namespace Sts2HumanRecorder;

internal sealed record StateMember(string Key, string Member, string Lifecycle);

internal static class ContentStateRegistry
{
    public const string Version = "sts2-v0.107.1-state-projection-1";

    private static readonly Dictionary<string, StateMember[]> RelicRuntimeMembers = new(StringComparer.Ordinal)
    {
        ["ArtOfWar"] = M(("any_attacks_played_last_turn", "AnyAttacksPlayedLastTurn", "turn"), ("any_attacks_played_this_turn", "AnyAttacksPlayedThisTurn", "turn")),
        ["BeatingRemnant"] = M(("damage_received_this_turn", "DamageReceivedThisTurn", "turn")),
        ["BeltBuckle"] = M(("dexterity_applied", "DexterityApplied", "combat")),
        ["BookOfFiveRings"] = M(("cards_added_since_last_trigger", "CardsAddedSinceLastTrigger", "run"), ("is_activating", "IsActivating", "run")),
        ["BrilliantScarf"] = M(("cards_played_this_turn", "CardsPlayedThisTurn", "turn")),
        ["BurningSticks"] = M(("was_used_this_combat", "WasUsedThisCombat", "combat")),
        ["CentennialPuzzle"] = M(("used_this_combat", "UsedThisCombat", "combat")),
        ["DiamondDiadem"] = M(("cards_played_this_turn", "CardsPlayedThisTurn", "turn")),
        ["EmotionChip"] = M(("lost_hp_in_previous_turn", "LostHpInPreviousTurn", "turn")),
        ["FakeOrichalcum"] = M(("should_trigger", "ShouldTrigger", "turn")),
        ["GalacticDust"] = M(("is_activating", "IsActivating", "run")),
        ["HappyFlower"] = M(("is_activating", "IsActivating", "run")),
        ["FakeHappyFlower"] = M(("is_activating", "IsActivating", "run")),
        ["IronClub"] = M(("is_activating", "IsActivating", "run")),
        ["JossPaper"] = M(("ethereal_count", "EtherealCount", "turn"), ("is_activating", "IsActivating", "run")),
        ["Kunai"] = M(("attacks_played_this_turn", "AttacksPlayedThisTurn", "turn"), ("is_activating", "IsActivating", "turn")),
        ["Kusarigama"] = M(("attacks_played_this_turn", "AttacksPlayedThisTurn", "turn"), ("is_activating", "IsActivating", "turn")),
        ["LastingCandy"] = M(("is_activating", "IsActivating", "combat"), ("is_in_triggering_combat", "IsInTriggeringCombat", "combat")),
        ["LetterOpener"] = M(("skills_played_this_turn", "SkillsPlayedThisTurn", "turn"), ("is_activating", "IsActivating", "turn")),
        ["Metronome"] = M(("orbs_channeled", "OrbsChanneled", "run"), ("is_activating", "IsActivating", "run")),
        ["MiniRegent"] = M(("used_this_turn", "UsedThisTurn", "turn")),
        ["MusicBox"] = M(("was_used_this_turn", "WasUsedThisTurn", "turn")),
        ["Orichalcum"] = M(("should_trigger", "ShouldTrigger", "turn")),
        ["OrnamentalFan"] = M(("attacks_played_this_turn", "AttacksPlayedThisTurn", "turn"), ("is_activating", "IsActivating", "turn")),
        ["PaelsEye"] = M(("used_this_combat", "UsedThisCombat", "combat"), ("was_owner_part_of_last_player_turn", "WasOwnerPartOfLastPlayerTurn", "turn")),
        ["PaelsLegion"] = M(("cooldown", "Cooldown", "combat"), ("triggered_block_last_turn", "TriggeredBlockLastTurn", "turn")),
        ["PaelsTears"] = M(("had_leftover_energy", "HadLeftoverEnergy", "turn")),
        ["PenNib"] = M(("is_activating", "IsActivating", "run")),
        ["Permafrost"] = M(("activated_this_combat", "ActivatedThisCombat", "combat")),
        ["Pocketwatch"] = M(("cards_played_last_turn", "_cardsPlayedLastTurn", "turn"), ("cards_played_this_turn", "_cardsPlayedThisTurn", "turn")),
        ["PollinousCore"] = M(("is_activating", "IsActivating", "run")),
        ["RainbowRing"] = M(("activation_count_this_turn", "ActivationCountThisTurn", "turn"), ("attacks_played_this_turn", "AttacksPlayedThisTurn", "turn"), ("powers_played_this_turn", "PowersPlayedThisTurn", "turn"), ("skills_played_this_turn", "SkillsPlayedThisTurn", "turn")),
        ["RedSkull"] = M(("strength_applied", "StrengthApplied", "combat")),
        ["RuinedHelmet"] = M(("used_this_combat", "UsedThisCombat", "combat")),
        ["Shuriken"] = M(("attacks_played_this_turn", "AttacksPlayedThisTurn", "turn"), ("is_activating", "IsActivating", "turn")),
        ["StoneCalendar"] = M(("is_activating", "IsActivating", "combat")),
        ["ThrowingAxe"] = M(("used_this_combat", "UsedThisCombat", "combat")),
        ["TuningFork"] = M(("is_activating", "IsActivating", "run")),
        ["UnsettlingLamp"] = M(("is_finished_triggering", "IsFinishedTriggering", "combat")),
        ["Vambrace"] = M(("block_gained_this_combat", "BlockGainedThisCombat", "combat")),
        ["VelvetChoker"] = M(("cards_played_this_turn", "_cardsPlayedThisTurn", "turn"), ("should_prevent_card_play", "ShouldPreventCardPlay", "turn"))
    };

    private static readonly Dictionary<string, StateMember[]> EnchantmentRuntimeMembers = new(StringComparer.Ordinal)
    {
        ["Glam"] = M(("used_this_combat", "UsedThisCombat", "combat")),
        ["Momentum"] = M(("extra_damage", "ExtraDamage", "combat"))
    };

    private static readonly Dictionary<string, StateMember[]> CardRuntimeMembers = new(StringComparer.Ordinal)
    {
        ["DeathsDoor"] = M(("doom_applied_this_turn", "WasDoomAppliedThisTurn", "turn")),
        ["EvilEye"] = M(("card_exhausted_this_turn", "WasCardExhaustedThisTurn", "turn")),
        ["Fetch"] = M(("played_this_turn", "HasBeenPlayedThisTurn", "turn")),
        ["Flatten"] = M(("osty_attacked_this_turn", "HasOstyAttackedThisTurn", "turn")),
        ["ForgottenRitual"] = M(("card_exhausted_this_turn", "WasCardExhaustedThisTurn", "turn")),
        ["Ftl"] = M(("can_draw_card", "CanDrawCard", "turn")),
        ["Normality"] = M(("cards_played_this_turn", "CardsPlayedThisTurn", "turn"), ("should_prevent_card_play", "ShouldPreventCardPlay", "turn")),
        ["PactsEnd"] = M(("can_deal_damage", "CanDealDamage", "turn")),
        ["Regret"] = M(("cards_in_hand", "CardsInHand", "turn")),
        ["Restlessness"] = M(("is_only_card_in_hand", "IsOnlyCardInHand", "turn")),
        ["SovereignBlade"] = M(("created_through_forge", "CreatedThroughForge", "combat"), ("current_damage", "CurrentDamage", "combat"), ("current_repeats", "CurrentRepeats", "combat")),
        ["Wither"] = M(("fake_upgrade_level", "FakeUpgradeLevel", "run"))
    };

    public static IReadOnlyList<StateMember> RelicRuntime(object relic) =>
        RelicRuntimeMembers.GetValueOrDefault(relic.GetType().Name) ?? Array.Empty<StateMember>();

    public static IReadOnlyList<StateMember> EnchantmentRuntime(object enchantment) =>
        EnchantmentRuntimeMembers.GetValueOrDefault(enchantment.GetType().Name) ?? Array.Empty<StateMember>();

    public static IReadOnlyList<StateMember> CardRuntime(object card) =>
        CardRuntimeMembers.GetValueOrDefault(card.GetType().Name) ?? Array.Empty<StateMember>();

    public static bool IsKnownBaseRelic(object relic) =>
        relic.GetType().Assembly == typeof(MegaCrit.Sts2.Core.Modding.ModInitializerAttribute).Assembly;

    public static string PersistentLifecycle(string modelType, string property)
    {
        if (property.Contains("ThisTurn", StringComparison.Ordinal) || property.Contains("LastTurn", StringComparison.Ordinal)
            || property.Contains("PreviousTurn", StringComparison.Ordinal)) return "turn";
        if (property.Contains("ThisCombat", StringComparison.Ordinal)) return "combat";
        if (modelType is "LavaLamp" && property == "TookDamageThisCombat") return "combat";
        return "run";
    }

    private static StateMember[] M(params (string Key, string Member, string Lifecycle)[] rows) =>
        rows.Select(row => new StateMember(row.Key, row.Member, row.Lifecycle)).ToArray();
}
