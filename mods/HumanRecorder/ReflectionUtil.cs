using System.Collections;
using System.Reflection;

namespace Sts2HumanRecorder;

internal static class ReflectionUtil
{
    private const BindingFlags Any = BindingFlags.Instance | BindingFlags.Static |
        BindingFlags.Public | BindingFlags.NonPublic;

    public static object? Get(object? value, params string[] names)
    {
        if (value is null) return null;
        var type = value as Type ?? value.GetType();
        var target = value is Type ? null : value;
        foreach (var name in names)
        {
            for (var cursor = type; cursor is not null; cursor = cursor.BaseType)
            {
                try
                {
                    var property = cursor.GetProperty(name, Any | BindingFlags.DeclaredOnly);
                    if (property is not null && property.GetIndexParameters().Length == 0)
                        return property.GetValue(target);
                    var field = cursor.GetField(name, Any | BindingFlags.DeclaredOnly);
                    if (field is not null) return field.GetValue(target);
                }
                catch { }
            }
        }
        return null;
    }

    public static object? Call(object? value, string method, params object?[] args)
    {
        if (value is null) return null;
        var type = value as Type ?? value.GetType();
        var target = value is Type ? null : value;
        try
        {
            var candidates = type.GetMethods(Any).Where(m => m.Name == method && m.GetParameters().Length == args.Length);
            foreach (var candidate in candidates)
            {
                try { return candidate.Invoke(target, args); } catch { }
            }
        }
        catch { }
        return null;
    }

    public static IEnumerable<object> Items(object? value)
    {
        if (value is null || value is string) yield break;
        if (value is IEnumerable enumerable)
            foreach (var item in enumerable)
                if (item is not null) yield return item;
    }

    public static string? Id(object? value)
    {
        if (value is null) return null;
        var id = Get(value, "Id", "ModelId", "OptionId", "TextKey");
        return id?.ToString() ?? value.ToString();
    }

    public static int Int(object? value, int fallback = 0)
    {
        try { return value is null ? fallback : Convert.ToInt32(value); } catch { return fallback; }
    }

    public static bool Bool(object? value, bool fallback = false)
    {
        try { return value is null ? fallback : Convert.ToBoolean(value); } catch { return fallback; }
    }
}
