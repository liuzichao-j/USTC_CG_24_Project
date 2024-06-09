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
uniform float arg2;
uniform float arg3;
uniform float arg4;

layout(location = 0) out vec4 Color;

const int kernelSize = 6;
const float PI = 3.14159265359;
float Gauss_Bilateral(float dist_sq, vec3 col1, vec3 col2)
{
    return exp(-arg2 * dist_sq - arg3 * dot(col1 - col2, col1 - col2));
}
float len(vec3 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

void main() 
{
    vec2 uv = gl_FragCoord.xy / iResolution;
    vec2 uv_base = min(iResolution.x, iResolution.y) / iResolution;
    float max_dist = arg1;
    vec2 idnh = max_dist * vec2(uv_base.x, 0.0) / float(kernelSize);
    vec2 idnv = max_dist * vec2(0.0, uv_base.y) / float(kernelSize);
    vec3 csrc = texture(baseColorSampler, uv).rgb;
    vec3 col = vec3(0.0, 0.0, 0.0);
    float sum = 0.0, dist, cur;
    for(int i = -kernelSize; i <= kernelSize; i++)
    {
        for(int j = -kernelSize; j <= kernelSize; j++)
        {
            vec3 cdest = texture(baseColorSampler, uv + float(i) * idnh + float(j) * idnv).rgb;
            cur = Gauss_Bilateral(float(i * i + j * j) / float(kernelSize * kernelSize), csrc, cdest);
            sum += cur;
            col += cdest * cur;
        }
    }
    col /= sum;
    Color = vec4(col, 1.0);
}