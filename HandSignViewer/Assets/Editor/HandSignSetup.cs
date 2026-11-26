using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// 장면 자동 생성 + 스크린샷 (메뉴 HandSign/..., 또는 batchmode -executeMethod).
/// </summary>
public static class HandSignSetup
{
    const string ScenePath = "Assets/Scenes/HandSignViewer.unity";

    [MenuItem("HandSign/Create Scene")]
    public static void CreateScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        var camGo = new GameObject("Main Camera");
        camGo.tag = "MainCamera";
        var cam = camGo.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.11f, 0.13f, 0.17f);
        cam.fieldOfView = 38f;
        cam.nearClipPlane = 0.01f;
        camGo.transform.position = new Vector3(0f, 0.0f, -0.78f);
        camGo.transform.LookAt(new Vector3(0f, 0.0f, 0f));

        var lightGo = new GameObject("Key Light");
        var light = lightGo.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.1f;
        lightGo.transform.rotation = Quaternion.Euler(35f, -25f, 0f);
        var fillGo = new GameObject("Fill Light");
        var fill = fillGo.AddComponent<Light>();
        fill.type = LightType.Directional;
        fill.intensity = 0.45f;
        fillGo.transform.rotation = Quaternion.Euler(-20f, 150f, 0f);
        RenderSettings.ambientLight = new Color(0.35f, 0.37f, 0.42f);

        var player = new GameObject("HandSignPlayer");
        player.AddComponent<HandSignPlayer>();

        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        Debug.Log($"[HandSign] scene saved: {ScenePath}");
    }

    /// <summary>
    /// batchmode 용: 장면 생성 → 지정 클립/프레임을 렌더해 PNG 저장.
    /// 인자: -captureDir <폴더>  -captures "clip:frame,clip:frame,..."
    /// </summary>
    public static void CaptureBatch()
    {
        string outDir = Arg("-captureDir") ?? Path.GetFullPath("Captures");
        string spec = Arg("-captures") ?? "0:0,0:10,0:20";
        Directory.CreateDirectory(outDir);

        CreateScene();
        var player = Object.FindFirstObjectByType<HandSignPlayer>();
        player.Init();
        var cam = Camera.main;
        var rt = new RenderTexture(1600, 900, 24) { antiAliasing = 8 };
        cam.targetTexture = rt;
        var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);

        foreach (var item in spec.Split(','))
        {
            var p = item.Split(':');
            int ci = int.Parse(p[0]), f = int.Parse(p[1]);
            if (ci >= player.ClipCount) continue;
            player.SetClipAndFrame(ci, f);
            cam.Render();
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            string path = Path.Combine(outDir, $"{player.ClipName(ci)}_f{f:D3}.png");
            File.WriteAllBytes(path, tex.EncodeToPNG());
            Debug.Log($"[HandSign] captured {path}");
        }
        RenderTexture.active = null;
        cam.targetTexture = null;
        // 장면은 CreateScene 에서 '빈 플레이어' 상태로 이미 저장됨.
        // 여기서 다시 저장하면 캡처용으로 만든 손 오브젝트가 장면에 남아 실행 시 중복되므로 저장하지 않는다.
    }

    static string Arg(string name)
    {
        var a = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < a.Length - 1; i++)
            if (a[i] == name) return a[i + 1];
        return null;
    }
}
