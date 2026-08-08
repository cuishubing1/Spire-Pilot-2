using Godot;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2HumanRecorder;

internal static class RecorderOverlay
{
    private const double RefreshSeconds = 1.0;
    private static CanvasLayer? _layer;
    private static Label? _contextLabel;
    private static Label? _statusDot;
    private static Label? _statusLabel;
    private static string? _lastRefreshError;
    private static string? _lastContextText;
    private static string? _lastStatusText;
    private static string? _lastStatusCode;

    public static void Install()
    {
        if (_layer is not null && GodotObject.IsInstanceValid(_layer)) return;
        if (Engine.GetMainLoop() is not SceneTree tree || tree.Root is null)
            throw new InvalidOperationException("Godot scene tree is not ready");

        _layer = new CanvasLayer { Name = "HumanRecorderOverlay", Layer = 120 };
        var panel = new PanelContainer
        {
            Name = "RecorderPanel",
            MouseFilter = Control.MouseFilterEnum.Ignore,
            AnchorLeft = 1f,
            AnchorRight = 1f,
            AnchorTop = 0.4f,
            AnchorBottom = 0.4f,
            OffsetLeft = -184f,
            OffsetRight = -14f,
            OffsetTop = -32f,
            OffsetBottom = 32f
        };
        var background = new StyleBoxFlat
        {
            BgColor = new Color(0.035f, 0.043f, 0.055f, 0.82f),
            BorderColor = new Color(1f, 1f, 1f, 0.10f),
            BorderWidthLeft = 1,
            BorderWidthTop = 1,
            BorderWidthRight = 1,
            BorderWidthBottom = 1,
            CornerRadiusTopLeft = 8,
            CornerRadiusTopRight = 8,
            CornerRadiusBottomLeft = 8,
            CornerRadiusBottomRight = 8,
            ContentMarginLeft = 12f,
            ContentMarginRight = 12f,
            ContentMarginTop = 8f,
            ContentMarginBottom = 8f
        };
        panel.AddThemeStyleboxOverride("panel", background);

        var rows = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        rows.AddThemeConstantOverride("separation", 3);
        _contextLabel = NewLabel(15, new Color(0.92f, 0.94f, 0.98f));
        var statusRow = new HBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        statusRow.AddThemeConstantOverride("separation", 6);
        _statusDot = NewLabel(14, Colors.White);
        _statusLabel = NewLabel(14, Colors.White);
        statusRow.AddChild(_statusDot);
        statusRow.AddChild(_statusLabel);
        rows.AddChild(_contextLabel);
        rows.AddChild(statusRow);
        panel.AddChild(rows);
        _layer.AddChild(panel);

        var timer = new Godot.Timer
        {
            Name = "RecorderRefreshTimer",
            WaitTime = RefreshSeconds,
            OneShot = false,
            Autostart = true,
            ProcessCallback = Godot.Timer.TimerProcessCallback.Idle
        };
        timer.Timeout += Refresh;
        _layer.AddChild(timer);
        tree.Root.AddChild(_layer);
        Refresh();
    }

    private static Label NewLabel(int size, Color color)
    {
        var label = new Label { MouseFilter = Control.MouseFilterEnum.Ignore };
        label.AddThemeFontSizeOverride("font_size", size);
        label.AddThemeColorOverride("font_color", color);
        label.AddThemeColorOverride("font_shadow_color", new Color(0f, 0f, 0f, 0.55f));
        label.AddThemeConstantOverride("shadow_offset_x", 1);
        label.AddThemeConstantOverride("shadow_offset_y", 1);
        return label;
    }

    private static void Refresh()
    {
        if (_contextLabel is null || _statusDot is null || _statusLabel is null) return;
        try
        {
            var (act, floor) = ReadGameContext();
            var contextText = $"Act {Display(act)}   层 {Display(floor)}";
            if (!string.Equals(_lastContextText, contextText, StringComparison.Ordinal))
            {
                _contextLabel.Text = contextText;
                _lastContextText = contextText;
            }
            var state = RecorderSession.GetUiState();
            var presentation = Present(state.Status);
            var statusText = state.WarningCount > 0 && state.Status == "warning"
                ? $"{presentation.Text} ({state.WarningCount})"
                : presentation.Text;
            if (!string.Equals(_lastStatusText, statusText, StringComparison.Ordinal))
            {
                _statusLabel.Text = statusText;
                _lastStatusText = statusText;
            }
            if (!string.Equals(_lastStatusCode, state.Status, StringComparison.Ordinal))
            {
                _statusDot.Text = "●";
                _statusDot.AddThemeColorOverride("font_color", presentation.Color);
                _statusLabel.AddThemeColorOverride("font_color", presentation.Color);
                _lastStatusCode = state.Status;
            }
            _lastRefreshError = null;
        }
        catch (Exception ex)
        {
            _statusDot.Text = "●";
            _statusLabel.Text = "状态不可用";
            _statusDot.AddThemeColorOverride("font_color", new Color("ff5d73"));
            _statusLabel.AddThemeColorOverride("font_color", new Color("ff5d73"));
            if (!string.Equals(_lastRefreshError, ex.Message, StringComparison.Ordinal))
            {
                _lastRefreshError = ex.Message;
                MainFile.Logger.Warn("HumanRecorder overlay refresh failed: " + ex.Message);
            }
        }
    }

    private static (int Act, int Floor) ReadGameContext()
    {
        var state = ReflectionUtil.Get(RunManager.Instance, "State");
        return state is null ? (0, 0) : (
            ReflectionUtil.Int(ReflectionUtil.Get(state, "CurrentActIndex")) + 1,
            ReflectionUtil.Int(ReflectionUtil.Get(state, "TotalFloor")));
    }

    private static string Display(int value) => value > 0 ? value.ToString() : "—";

    internal static RecorderStatusPresentation Present(string status) => status switch
    {
        "recording" => new("记录中", new Color("59d98e")),
        "restoring" => new("恢复中", new Color("ffd166")),
        "warning" => new("记录有警告", new Color("f4a261")),
        "error" => new("记录异常", new Color("ff5d73")),
        _ => new("待机", new Color("aab3c2"))
    };
}

internal readonly record struct RecorderStatusPresentation(string Text, Color Color);
