// Left empty. This is optional. For implemetation, you can find many references from nearby shaders. You might need random number generators (RNG) to distribute points in a (Hemi)sphere. You can ask AI for both of them (RNG and sampling in a sphere) or try to find some resources online. Later I will add some links to the document about this.
#version 430 core

uniform vec2 iResolution;

uniform sampler2D positionSampler;
uniform sampler2D normalSampler;
uniform sampler2D depthSampler;
uniform sampler2D baseColorSampler;

uniform mat4 projection;
uniform mat4 view;
uniform float arg1;

layout(location = 0) out vec4 Color;

const int kernelSize = 8;
const float PI = 3.14159265359;

//copied and adjusted from https://zhuanlan.zhihu.com/p/599263679
vec3 hash33(vec3 p3)
{
    p3 = fract(p3 * vec3(1.4031, 2.1060, 2.0973) * vec3(50.0)); 
    p3 += dot(p3, p3.yxz + 36.33); 
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}
    
vec3 hash33_sphere(vec3 p3)
{
    vec3 hash = hash33(p3);
    hash.xy *= 2 * PI;
    return vec3(vec2(cos(hash.x), sin(hash.x)) * sin(hash.y), cos(hash.y))
             * pow(hash.z, 0.333);
}

// https://stackoverflow.com/questions/9446888/best-way-to-detect-nans-in-opengl-shaderms
bool isnan( float val )
{
  return ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
  // important: some nVidias failed to cope with version below.
  // Probably wrong optimization.
  /*return ( val <= 0.0 || 0.0 <= val ) ? false : true;*/
}

float len(vec3 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

void main() 
{
    const float uv_scale = 1.0;
    vec2 uv = gl_FragCoord.xy / iResolution * uv_scale + (1.0 - uv_scale) * 0.5;
    vec3 frag_pos = texture(positionSampler, uv).xyz;
    vec3 normal = texture(normalSampler, uv).xyz;
    vec4 frag_pos_clip = projection * view * vec4(frag_pos, 1.0);
    float depth = frag_pos_clip.z / frag_pos_clip.w;
    
    if(len(normal) < 0.1)
    {
        //background
        // Color = base_color;
        Color = vec4(1.0);
        return;
    }

    float occlusion = 0.0;
    for(int i = 0; i < kernelSize; i++)
    {
        vec3 dir = hash33_sphere(vec3(uv, i)); 
        if(dot(dir, normal) < 0.0) dir = -dir;
        vec3 samp = frag_pos + dir * arg1;
        vec4 samp_pos_clip = projection * view * vec4(samp, 1.0);
        vec3 samp_pos_camera_space = samp_pos_clip.xyz / samp_pos_clip.w;
        float samp_depth = samp_pos_camera_space.z;
        vec2 samp_uv = samp_pos_camera_space.xy * 0.5 + 0.5;
        vec3 samp_pos = texture(positionSampler, samp_uv).xyz;

        // 剔除背景
        vec3 frag_norm = texture(normalSampler, samp_uv).rgb;
        if(len(frag_norm) < 0.1)
            continue;

        float frag_depth = texture(depthSampler, samp_uv).r;

        float rangeCheck = smoothstep(0.0, 1.0, arg1 / (len(frag_pos - samp_pos)));
        occlusion += (samp_depth > frag_depth ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion = 1.0 - occlusion / kernelSize;
    // occlusion = pow(occlusion, 1.0 / 2.2);

    // Trick: 纯视觉经验地把 ssao 调到一个更明显的值，而不是采用 gamma 校正
    occlusion = occlusion * occlusion * occlusion * occlusion;

    Color.rgb = vec3(occlusion);
    Color.a = 1.0;
}