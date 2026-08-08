using System.Runtime.CompilerServices;
using System.Text.Json;

namespace Sts2HumanRecorder;

internal sealed record CardIdentityValue(string LineageId, string Quality);

internal static class CardIdentity
{
    private sealed class Holder(string id) { public string Id { get; } = id; }

    private static readonly object Gate = new();
    private static readonly ConditionalWeakTable<object, Holder> Objects = new();
    private static Dictionary<string, List<string>> _previousDeckSlots = new(StringComparer.Ordinal);
    private static Dictionary<string, List<string>>? _currentDeckSlots;
    private static readonly Dictionary<string, int> CurrentOccurrences = new(StringComparer.Ordinal);

    public static void ResetForRun()
    {
        lock (Gate)
        {
            _previousDeckSlots = new Dictionary<string, List<string>>(StringComparer.Ordinal);
            _currentDeckSlots = null;
            CurrentOccurrences.Clear();
            // ConditionalWeakTable cannot be cleared on older targets, but replacing
            // cross-run deck slots is sufficient: live objects from a completed run
            // are never reused by the next RunManager state.
        }
    }

    public static void BeginDeckSnapshot()
    {
        lock (Gate)
        {
            _currentDeckSlots = new Dictionary<string, List<string>>(StringComparer.Ordinal);
            CurrentOccurrences.Clear();
        }
    }

    public static void EndDeckSnapshot()
    {
        lock (Gate)
        {
            if (_currentDeckSlots is not null) _previousDeckSlots = _currentDeckSlots;
            _currentDeckSlots = null;
            CurrentOccurrences.Clear();
        }
    }

    public static CardIdentityValue Resolve(object card, bool deckCard)
    {
        lock (Gate)
        {
            var signature = Signature(card);
            var occurrence = 0;
            if (deckCard)
            {
                CurrentOccurrences.TryGetValue(signature, out occurrence);
                CurrentOccurrences[signature] = occurrence + 1;
            }

            if (Objects.TryGetValue(card, out var existing))
            {
                if (deckCard) AddCurrent(signature, existing.Id);
                return new CardIdentityValue(existing.Id, "stable_object");
            }

            string id;
            string quality;
            if (deckCard && _previousDeckSlots.TryGetValue(signature, out var prior) && occurrence < prior.Count)
            {
                id = prior[occurrence];
                quality = prior.Count == 1 ? "reconciled_exact" : "reconciled_ambiguous_duplicate";
            }
            else
            {
                id = "card-" + Guid.NewGuid().ToString("N");
                quality = deckCard ? "new_deck_card" : "new_runtime_card";
            }
            Objects.Add(card, new Holder(id));
            if (deckCard) AddCurrent(signature, id);
            return new CardIdentityValue(id, quality);
        }
    }

    private static void AddCurrent(string signature, string id)
    {
        if (_currentDeckSlots is null) return;
        if (!_currentDeckSlots.TryGetValue(signature, out var list))
            _currentDeckSlots[signature] = list = new List<string>();
        list.Add(id);
    }

    private static string Signature(object card)
    {
        var enchantment = ReflectionUtil.Get(card, "Enchantment");
        var affliction = ReflectionUtil.Get(card, "Affliction");
        return JsonSerializer.Serialize(new object?[]
        {
            ReflectionUtil.Id(card),
            ReflectionUtil.Int(ReflectionUtil.Get(card, "CurrentUpgradeLevel")),
            ReflectionUtil.Id(enchantment), ReflectionUtil.Get(enchantment, "Amount")?.ToString(),
            ReflectionUtil.Id(affliction), ReflectionUtil.Get(affliction, "Amount")?.ToString()
        });
    }
}
