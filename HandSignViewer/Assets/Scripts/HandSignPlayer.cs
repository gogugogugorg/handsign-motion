using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

/// <summary>
/// p7_export_unity_hands.py 가 만든 CSV(원본 / 후처리)를 불러와 두 손을 나란히 재생한다.
///   왼쪽 = 원본 (MediaPipe 그대로, 검출 안 된 손은 안 보임)
///   오른쪽 = 후처리 (보간 / 팔 기반 추정으로 채운 손은 노란색)
/// 조작: ← / → 클립 변경, Space 일시정지, , / . 한 프레임 이동
/// </summary>
public class HandSignPlayer : MonoBehaviour
{
    [Tooltip("StreamingAssets 아래 폴더")]
    public string folder = "HandSign";
    public float groupSpacing = 0.52f;
    public float playbackSpeed = 1f;

    public static readonly Color RawColor = new Color(0.96f, 0.45f, 0.19f);    // 주황
    public static readonly Color CleanColor = new Color(0.0f, 0.63f, 0.50f);   // 청록
    public static readonly Color FilledColor = new Color(0.95f, 0.75f, 0.10f); // 노랑

    class Clip
    {
        public string name;
        public float[] times;
        public Vector3[,] raw, clean;     // [frame, 42]
        public int[,] rawState, cleanState; // [frame, 2] 0 없음, 1 검출, 2 채움
        public int Frames => times.Length;
    }

    readonly List<Clip> clips = new List<Clip>();
    HandModel[] rawHands, cleanHands;
    TextMesh titleText, rawLabel, cleanLabel, rawStatus, cleanStatus, legendText;
    int clipIdx, frame;
    float clipTime;
    bool paused;

    public int ClipCount => clips.Count;
    public string ClipName(int i) => clips[i].name;
    public int FrameCount(int i) => clips[i].Frames;

    void Start()
    {
        Init();
        ShowFrame(0);
    }

    public void Init()
    {
        LoadClips();
        var shader = Shader.Find("Standard");
        rawHands = new[] { MakeHand("Raw_L", shader), MakeHand("Raw_R", shader) };
        cleanHands = new[] { MakeHand("Clean_L", shader), MakeHand("Clean_R", shader) };
        foreach (var h in rawHands) h.transform.localPosition = new Vector3(-groupSpacing / 2, 0, 0);
        foreach (var h in cleanHands) h.transform.localPosition = new Vector3(groupSpacing / 2, 0, 0);

        rawLabel = MakeText("RAW  (MediaPipe)", new Vector3(-groupSpacing / 2, 0.145f, 0), 0.032f, RawColor);
        cleanLabel = MakeText("POST-PROCESSED", new Vector3(groupSpacing / 2, 0.145f, 0), 0.032f, CleanColor);
        rawStatus = MakeText("", new Vector3(-groupSpacing / 2, -0.14f, 0), 0.022f, new Color(0.85f, 0.85f, 0.85f));
        cleanStatus = MakeText("", new Vector3(groupSpacing / 2, -0.14f, 0), 0.022f, new Color(0.85f, 0.85f, 0.85f));
        titleText = MakeText("", new Vector3(0, 0.19f, 0), 0.024f, Color.white);
        legendText = MakeText("yellow = filled by interpolation / arm estimate      missing hand = not drawn",
                              new Vector3(0, -0.175f, 0), 0.018f, new Color(0.7f, 0.7f, 0.7f));
    }

    HandModel MakeHand(string name, Shader shader)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        var h = go.AddComponent<HandModel>();
        h.Build(shader);
        return h;
    }

    TextMesh MakeText(string text, Vector3 pos, float size, Color color)
    {
        var go = new GameObject("Label");
        go.transform.SetParent(transform, false);
        go.transform.localPosition = pos;
        var tm = go.AddComponent<TextMesh>();
        var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        tm.font = font;
        go.GetComponent<MeshRenderer>().sharedMaterial = font.material;
        tm.text = text;
        tm.fontSize = 64;
        tm.characterSize = size * 0.15f;
        tm.anchor = TextAnchor.MiddleCenter;
        tm.alignment = TextAlignment.Center;
        tm.color = color;
        return tm;
    }

    // ------------------------------------------------------------
    // CSV
    // ------------------------------------------------------------
    void LoadClips()
    {
        clips.Clear();
        string dir = Path.Combine(Application.streamingAssetsPath, folder);
        if (!Directory.Exists(dir))
        {
            Debug.LogError($"[HandSign] folder not found: {dir}  (run p7_export_unity_hands.py)");
            return;
        }
        foreach (var rawPath in Directory.GetFiles(dir, "*_raw.csv"))
        {
            string name = Path.GetFileName(rawPath).Replace("_raw.csv", "");
            string cleanPath = Path.Combine(dir, name + "_clean.csv");
            if (!File.Exists(cleanPath)) continue;
            var c = new Clip { name = name };
            ReadCsv(rawPath, out c.times, out c.raw, out c.rawState);
            ReadCsv(cleanPath, out _, out c.clean, out c.cleanState);
            clips.Add(c);
        }
        // 예시 클립(boar_20261127_015305)을 맨 앞으로
        clips.Sort((a, b) => a.name == "boar_20261127_015305" ? -1 : b.name == "boar_20261127_015305" ? 1
                                                            : string.CompareOrdinal(a.name, b.name));
        Debug.Log($"[HandSign] loaded {clips.Count} clips from {dir}");
    }

    static void ReadCsv(string path, out float[] times, out Vector3[,] pts, out int[,] state)
    {
        var lines = File.ReadAllLines(path);
        int n = (lines.Length - 1) / 42;
        times = new float[n];
        pts = new Vector3[n, 42];
        state = new int[n, 2];
        var ci = CultureInfo.InvariantCulture;
        for (int i = 1; i < lines.Length; i++)
        {
            var v = lines[i].Split(',');
            if (v.Length < 7) continue;
            int f = int.Parse(v[0], ci), j = int.Parse(v[2], ci);
            if (f >= n) continue;
            times[f] = float.Parse(v[1], ci);
            pts[f, j] = new Vector3(float.Parse(v[3], ci), float.Parse(v[4], ci), float.Parse(v[5], ci));
            state[f, j / 21] = int.Parse(v[6], ci);
        }
    }

    // ------------------------------------------------------------
    // 재생
    // ------------------------------------------------------------
    void Update()
    {
        if (clips.Count == 0) return;
        if (Input.GetKeyDown(KeyCode.RightArrow)) SelectClip(clipIdx + 1);
        if (Input.GetKeyDown(KeyCode.LeftArrow)) SelectClip(clipIdx - 1);
        if (Input.GetKeyDown(KeyCode.Space)) paused = !paused;
        if (Input.GetKeyDown(KeyCode.Period)) { paused = true; StepFrame(1); }
        if (Input.GetKeyDown(KeyCode.Comma)) { paused = true; StepFrame(-1); }
        if (paused) return;

        var c = clips[clipIdx];
        clipTime += Time.deltaTime * playbackSpeed;
        float end = c.times[c.Frames - 1] + 0.6f;      // 끝에서 잠깐 멈췄다가 반복
        if (clipTime > end) clipTime = 0f;
        int f = 0;
        while (f + 1 < c.Frames && c.times[f + 1] <= clipTime) f++;
        if (f != frame) ShowFrame(f);
    }

    void StepFrame(int d)
    {
        var c = clips[clipIdx];
        int f = (frame + d + c.Frames) % c.Frames;
        clipTime = c.times[f];
        ShowFrame(f);
    }

    public void SelectClip(int i)
    {
        if (clips.Count == 0) return;
        clipIdx = (i + clips.Count) % clips.Count;
        clipTime = 0f;
        ShowFrame(0);
    }

    public void ShowFrame(int f)
    {
        if (clips.Count == 0) return;
        var c = clips[clipIdx];
        frame = Mathf.Clamp(f, 0, c.Frames - 1);
        var buf = new Vector3[21];
        string[] side = { "L", "R" };
        string rs = "", cs = "";
        for (int h = 0; h < 2; h++)
        {
            int sr = c.rawState[frame, h], sc = c.cleanState[frame, h];
            for (int j = 0; j < 21; j++) buf[j] = c.raw[frame, h * 21 + j];
            rawHands[h].SetVisible(sr > 0);
            if (sr > 0) { rawHands[h].SetPose(buf); rawHands[h].SetColor(RawColor); }

            for (int j = 0; j < 21; j++) buf[j] = c.clean[frame, h * 21 + j];
            cleanHands[h].SetVisible(sc > 0);
            if (sc > 0) { cleanHands[h].SetPose(buf); cleanHands[h].SetColor(sc == 2 ? FilledColor : CleanColor); }

            rs += $"{side[h]}: {(sr > 0 ? "detected" : "MISSING")}   ";
            cs += $"{side[h]}: {(sc == 1 ? "detected" : sc == 2 ? "FILLED" : "missing")}   ";
        }
        rawStatus.text = rs.Trim();
        cleanStatus.text = cs.Trim();
        titleText.text = $"{c.name}    frame {frame + 1}/{c.Frames}    ({clipIdx + 1}/{clips.Count})";
    }

    public void SetClipAndFrame(int clip, int f)
    {
        clipIdx = Mathf.Clamp(clip, 0, clips.Count - 1);
        ShowFrame(f);
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 600, 22), "← / → : clip    Space : pause    , / . : step frame");
    }
}
