using UnityEngine;

/// <summary>
/// MediaPipe 손 21점으로 만드는 도형 손 모델.
/// 관절 = 구, 뼈 = 원기둥, 손바닥 = 면(메시). 매 프레임 SetPose 로 위치를 갱신한다.
/// </summary>
public class HandModel : MonoBehaviour
{
    // MediaPipe 손 연결 (손목 0, 엄지 1-4, 검지 5-8, 중지 9-12, 약지 13-16, 새끼 17-20)
    static readonly int[,] Bones =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 },
        { 0, 5 }, { 5, 6 }, { 6, 7 }, { 7, 8 },
        { 9, 10 }, { 10, 11 }, { 11, 12 },
        { 13, 14 }, { 14, 15 }, { 15, 16 },
        { 0, 17 }, { 17, 18 }, { 18, 19 }, { 19, 20 },
        { 5, 9 }, { 9, 13 }, { 13, 17 },
    };

    // 손바닥 면: 손목에서 부채꼴로 (0-1-5, 0-5-9, 0-9-13, 0-13-17)
    static readonly int[] PalmFan = { 0, 1, 5, 0, 5, 9, 0, 9, 13, 0, 13, 17 };

    Transform[] joints;
    Transform[] bones;
    Mesh palmMesh;
    Renderer[] renderers;
    Material skin, palm;

    /// <summary>관절 굵기: 손목 > 마디 > 손끝 순으로 가늘게</summary>
    static float JointRadius(int j)
    {
        if (j == 0) return 0.013f;
        int k = (j - 1) % 4;                 // 0 = 뿌리, 3 = 손끝
        float baseR = j <= 4 ? 0.0115f : 0.0105f;   // 엄지는 조금 굵게
        return baseR - k * 0.0012f;
    }

    public void Build(Shader shader)
    {
        skin = new Material(shader);
        palm = new Material(shader);
        skin.SetFloat("_Glossiness", 0.35f);
        palm.SetFloat("_Glossiness", 0.2f);

        joints = new Transform[21];
        for (int j = 0; j < 21; j++)
        {
            var s = CreatePart(PrimitiveType.Sphere, $"joint_{j}");
            float d = JointRadius(j) * 2f;
            s.localScale = new Vector3(d, d, d);
            joints[j] = s;
        }

        bones = new Transform[Bones.GetLength(0)];
        for (int b = 0; b < bones.Length; b++)
            bones[b] = CreatePart(PrimitiveType.Cylinder, $"bone_{Bones[b, 0]}_{Bones[b, 1]}");

        var palmGo = new GameObject("palm");
        palmGo.transform.SetParent(transform, false);
        palmMesh = new Mesh { name = "palm" };
        palmGo.AddComponent<MeshFilter>().sharedMesh = palmMesh;
        var pr = palmGo.AddComponent<MeshRenderer>();
        pr.sharedMaterial = palm;

        renderers = GetComponentsInChildren<Renderer>(true);
    }

    Transform CreatePart(PrimitiveType type, string name)
    {
        var go = GameObject.CreatePrimitive(type);
        go.name = name;
        Object.DestroyImmediate(go.GetComponent<Collider>());
        go.GetComponent<Renderer>().sharedMaterial = skin;
        go.transform.SetParent(transform, false);
        return go.transform;
    }

    public void SetColor(Color c)
    {
        skin.color = c;
        palm.color = Color.Lerp(c, Color.white, 0.25f);
    }

    public void SetVisible(bool v)
    {
        foreach (var r in renderers) r.enabled = v;
    }

    /// <summary>pts: 이 손의 21개 관절 (로컬 좌표, 미터)</summary>
    public void SetPose(Vector3[] pts)
    {
        for (int j = 0; j < 21; j++) joints[j].localPosition = pts[j];

        for (int b = 0; b < bones.Length; b++)
        {
            Vector3 a = pts[Bones[b, 0]], c = pts[Bones[b, 1]];
            Vector3 d = c - a;
            float len = d.magnitude;
            var t = bones[b];
            if (len < 1e-5f) { t.localScale = Vector3.zero; continue; }
            // 원기둥 기본 높이 2 (y -1~1), 지름 1 → 두 관절 사이에 맞춤
            float r = Mathf.Min(JointRadius(Bones[b, 0]), JointRadius(Bones[b, 1])) * 0.85f;
            t.localPosition = (a + c) * 0.5f;
            t.localRotation = Quaternion.FromToRotation(Vector3.up, d / len);
            t.localScale = new Vector3(r * 2f, len * 0.5f, r * 2f);
        }

        // 손바닥: 양면이 보이도록 앞/뒤 삼각형 둘 다
        var v = new Vector3[PalmFan.Length * 2];
        var tri = new int[PalmFan.Length * 2];
        for (int i = 0; i < PalmFan.Length; i++)
        {
            v[i] = pts[PalmFan[i]];
            v[i + PalmFan.Length] = pts[PalmFan[i]];
        }
        for (int i = 0; i < PalmFan.Length; i += 3)
        {
            tri[i] = i; tri[i + 1] = i + 1; tri[i + 2] = i + 2;
            int o = PalmFan.Length;
            tri[o + i] = o + i; tri[o + i + 1] = o + i + 2; tri[o + i + 2] = o + i + 1;
        }
        palmMesh.Clear();
        palmMesh.vertices = v;
        palmMesh.triangles = tri;
        palmMesh.RecalculateNormals();
        palmMesh.RecalculateBounds();
    }
}
