from __future__ import annotations

from dataclasses import dataclass


POTION_CATALOG_VERSION = "combat-potion-catalog-0.1.0"
GAME_VERSION = "0.107.1"
STEAM_BUILD = 23811903

VALID_RARITIES = {"common", "uncommon", "rare", "special"}
VALID_POOLS = {"shared", "ironclad", "shared_special"}
VALID_EVALUATORS = {
    "paired_turn_boundary",
    "nested_choice",
    "sampled_rollout",
    "persistent_rollout",
    "upper_resource",
    "passive_reserve",
}
VALID_CHOICE_MODES = {
    "none",
    "random_offer_one",
    "random_generation",
    "single_hand",
    "single_draw",
    "single_discard",
    "hand_subset",
    "delayed_next_card",
    "automatic",
}


@dataclass(frozen=True)
class PotionSpec:
    potion_id: str
    title_zh: str
    rarity: str
    pool: str
    evaluator: str
    effect_tags: tuple[str, ...]
    choice_mode: str = "none"
    stochastic: bool = False
    horizon: str = "turn_boundary"
    upper_arbitration: bool = True
    setup: str = "default"

    def to_dict(self) -> dict[str, object]:
        return {
            "potion_id": self.potion_id,
            "title_zh": self.title_zh,
            "rarity": self.rarity,
            "pool": self.pool,
            "evaluator": self.evaluator,
            "effect_tags": list(self.effect_tags),
            "choice_mode": self.choice_mode,
            "stochastic": self.stochastic,
            "horizon": self.horizon,
            "upper_arbitration": self.upper_arbitration,
            "setup": self.setup,
        }


def _p(
    entry: str,
    title_zh: str,
    rarity: str,
    evaluator: str,
    *effect_tags: str,
    pool: str = "shared",
    choice_mode: str = "none",
    stochastic: bool = False,
    horizon: str = "turn_boundary",
    upper_arbitration: bool = True,
    setup: str = "default",
) -> PotionSpec:
    return PotionSpec(
        potion_id=f"POTION.{entry}",
        title_zh=title_zh,
        rarity=rarity,
        pool=pool,
        evaluator=evaluator,
        effect_tags=tuple(effect_tags),
        choice_mode=choice_mode,
        stochastic=stochastic,
        horizon=horizon,
        upper_arbitration=upper_arbitration,
        setup=setup,
    )


# Version-locked to the local v0.107.1 game assembly.  The catalog contains
# every normal shared potion, the three Ironclad potions, and shared special
# potions.  Other characters' exclusive potions are intentionally excluded.
POTION_SPECS: tuple[PotionSpec, ...] = (
    # Common shared.
    _p("ATTACK_POTION", "攻击药水", "common", "nested_choice", "generate_card", "attack", choice_mode="random_offer_one", stochastic=True),
    _p("BLOCK_POTION", "格挡药水", "common", "paired_turn_boundary", "block"),
    _p("COLORLESS_POTION", "无色药水", "common", "nested_choice", "generate_card", "colorless", choice_mode="random_offer_one", stochastic=True),
    _p("DEXTERITY_POTION", "敏捷药水", "common", "persistent_rollout", "dexterity", horizon="combat_end"),
    _p("ENERGY_POTION", "能量药水", "common", "paired_turn_boundary", "energy"),
    _p("EXPLOSIVE_AMPOULE", "爆炸安瓿", "common", "paired_turn_boundary", "damage", "all_enemies"),
    _p("FIRE_POTION", "火焰药水", "common", "paired_turn_boundary", "damage", "single_enemy"),
    _p("FLEX_POTION", "肌肉药水", "common", "paired_turn_boundary", "strength", "turn_only"),
    _p("POWER_POTION", "能力药水", "common", "nested_choice", "generate_card", "power", choice_mode="random_offer_one", stochastic=True, horizon="combat_end"),
    _p("SKILL_POTION", "技能药水", "common", "nested_choice", "generate_card", "skill", choice_mode="random_offer_one", stochastic=True),
    _p("SPEED_POTION", "速度药水", "common", "paired_turn_boundary", "dexterity", "turn_only"),
    _p("STRENGTH_POTION", "力量药水", "common", "persistent_rollout", "strength", horizon="combat_end"),
    _p("SWIFT_POTION", "迅捷药水", "common", "sampled_rollout", "draw", choice_mode="random_generation", stochastic=True),
    _p("VULNERABLE_POTION", "易伤药水", "common", "persistent_rollout", "vulnerable", "single_enemy", horizon="multi_turn"),
    _p("WEAK_POTION", "虚弱药水", "common", "persistent_rollout", "weak", "single_enemy", horizon="multi_turn"),
    # Common Ironclad.
    _p("BLOOD_POTION", "鲜血药水", "common", "upper_resource", "heal", pool="ironclad", horizon="run_state"),

    # Uncommon shared.
    _p("BLESSING_OF_THE_FORGE", "熔炉的祝福", "uncommon", "persistent_rollout", "upgrade_hand", horizon="combat_end"),
    _p("CLARITY", "明晰提取物", "uncommon", "sampled_rollout", "draw", "future_draw", choice_mode="random_generation", stochastic=True, horizon="multi_turn"),
    _p("CURE_ALL", "痊愈药水", "uncommon", "sampled_rollout", "energy", "draw", choice_mode="random_generation", stochastic=True),
    _p("DUPLICATOR", "复制药水", "uncommon", "nested_choice", "duplicate_next_card", choice_mode="delayed_next_card"),
    _p("FORTIFIER", "固化药水", "uncommon", "paired_turn_boundary", "multiply_block", setup="block_in_hand"),
    _p("FYSH_OIL", "异鱼之油", "uncommon", "persistent_rollout", "strength", "dexterity", horizon="combat_end"),
    _p("GAMBLERS_BREW", "赌徒特酿", "uncommon", "nested_choice", "discard", "draw", choice_mode="hand_subset", stochastic=True),
    _p("HEART_OF_IRON", "铁心药水", "uncommon", "persistent_rollout", "plating", horizon="combat_end"),
    _p("LIQUID_BRONZE", "流动铜液", "uncommon", "persistent_rollout", "thorns", horizon="combat_end"),
    _p("POTION_OF_BINDING", "缚魂药水", "uncommon", "persistent_rollout", "weak", "vulnerable", "all_enemies", horizon="multi_turn"),
    _p("POWDERED_DEMISE", "消亡粉末", "uncommon", "persistent_rollout", "end_turn_hp_loss", "single_enemy", horizon="combat_end"),
    _p("RADIANT_TINCTURE", "明耀酊剂", "uncommon", "persistent_rollout", "energy", "future_energy", horizon="multi_turn"),
    _p("REGEN_POTION", "再生药水", "uncommon", "persistent_rollout", "heal_over_time", horizon="combat_end"),
    _p("STABLE_SERUM", "稳定血清", "uncommon", "persistent_rollout", "retain_hand", horizon="multi_turn"),
    _p("TOUCH_OF_INSANITY", "癫狂之触", "uncommon", "nested_choice", "make_card_free", choice_mode="single_hand", horizon="combat_end"),
    # Uncommon Ironclad.
    _p("ASHWATER", "灰水", "uncommon", "nested_choice", "exhaust", pool="ironclad", choice_mode="hand_subset", horizon="combat_end"),

    # Rare shared.
    _p("BEETLE_JUICE", "甲虫汁", "rare", "persistent_rollout", "incoming_damage_reduction", horizon="multi_turn"),
    _p("BOTTLED_POTENTIAL", "瓶装潜能", "rare", "sampled_rollout", "shuffle_all", "draw", choice_mode="random_generation", stochastic=True),
    _p("DISTILLED_CHAOS", "精炼混沌", "rare", "sampled_rollout", "autoplay_draw_pile", choice_mode="automatic", stochastic=True),
    _p("DROPLET_OF_PRECOGNITION", "预知之滴", "rare", "nested_choice", "draw_pile_tutor", choice_mode="single_draw"),
    _p("ENTROPIC_BREW", "混沌药水", "rare", "sampled_rollout", "generate_potions", "inventory", choice_mode="random_generation", stochastic=True, horizon="run_state"),
    _p("FAIRY_IN_A_BOTTLE", "瓶中精灵", "rare", "passive_reserve", "automatic_revive", choice_mode="automatic", horizon="run_state"),
    _p("FRUIT_JUICE", "果汁", "rare", "upper_resource", "max_hp", horizon="run_state"),
    _p("GIGANTIFICATION_POTION", "超巨化药水", "rare", "nested_choice", "triple_next_attack", choice_mode="delayed_next_card"),
    _p("LIQUID_MEMORIES", "液态记忆", "rare", "nested_choice", "discard_tutor", "make_card_free", choice_mode="single_discard", setup="discard_card"),
    _p("LUCKY_TONIC", "幸运补剂", "rare", "persistent_rollout", "buffer", horizon="combat_end"),
    _p("MAZALETHS_GIFT", "马萨雷斯的赠礼", "rare", "persistent_rollout", "ritual", horizon="combat_end"),
    _p("OROBIC_ACID", "欧洛巴斯之酸", "rare", "sampled_rollout", "generate_card", "attack", "skill", "power", choice_mode="random_generation", stochastic=True),
    _p("SHACKLING_POTION", "镣铐药水", "rare", "paired_turn_boundary", "strength_reduction", "all_enemies", "turn_only"),
    _p("SHIP_IN_A_BOTTLE", "瓶中船", "rare", "persistent_rollout", "block", "future_block", horizon="multi_turn"),
    _p("SNECKO_OIL", "异蛇之油", "rare", "sampled_rollout", "draw", "randomize_cost", choice_mode="random_generation", stochastic=True),
    # Rare Ironclad.
    _p("SOLDIERS_STEW", "士兵炖汤", "rare", "persistent_rollout", "strike_replay", pool="ironclad", horizon="combat_end"),

    # Shared special potions present in the v0.107.1 assembly.
    _p("FOUL_POTION", "污浊药水", "special", "upper_resource", "damage", "all_creatures", "merchant_gold", pool="shared_special", horizon="run_state"),
    _p("GLOWWATER_POTION", "发光水", "special", "sampled_rollout", "exhaust_hand", "draw", pool="shared_special", choice_mode="random_generation", stochastic=True),
    _p("POTION_SHAPED_ROCK", "药水形状的石头", "special", "paired_turn_boundary", "damage", "single_enemy", pool="shared_special"),
)


POTION_SPECS_BY_ID = {spec.potion_id: spec for spec in POTION_SPECS}


def validate_potion_catalog() -> dict[str, int]:
    ids = [spec.potion_id for spec in POTION_SPECS]
    if len(ids) != len(set(ids)):
        raise ValueError("potion catalog contains duplicate ids")
    if len(POTION_SPECS) != 51:
        raise ValueError(f"expected 51 Ironclad-applicable potions, got {len(POTION_SPECS)}")
    for spec in POTION_SPECS:
        if not spec.potion_id.startswith("POTION."):
            raise ValueError(f"invalid potion id: {spec.potion_id}")
        if spec.rarity not in VALID_RARITIES:
            raise ValueError(f"invalid rarity for {spec.potion_id}: {spec.rarity}")
        if spec.pool not in VALID_POOLS:
            raise ValueError(f"invalid pool for {spec.potion_id}: {spec.pool}")
        if spec.evaluator not in VALID_EVALUATORS:
            raise ValueError(f"invalid evaluator for {spec.potion_id}: {spec.evaluator}")
        if spec.choice_mode not in VALID_CHOICE_MODES:
            raise ValueError(f"invalid choice mode for {spec.potion_id}: {spec.choice_mode}")
        if not spec.effect_tags or len(spec.effect_tags) != len(set(spec.effect_tags)):
            raise ValueError(f"invalid effect tags for {spec.potion_id}")
    return {
        "total": len(POTION_SPECS),
        "shared": sum(spec.pool == "shared" for spec in POTION_SPECS),
        "ironclad": sum(spec.pool == "ironclad" for spec in POTION_SPECS),
        "shared_special": sum(spec.pool == "shared_special" for spec in POTION_SPECS),
        "stochastic": sum(spec.stochastic for spec in POTION_SPECS),
    }
