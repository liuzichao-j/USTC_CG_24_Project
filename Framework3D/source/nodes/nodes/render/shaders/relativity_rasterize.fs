#version 430

layout(location = 0) out vec3 position;
layout(location = 1) out float depth;
layout(location = 2) out vec2 texcoords;
layout(location = 3) out vec3 diffuseColor;
layout(location = 4) out vec2 metallicRoughness;
layout(location = 5) out vec3 normal;

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vTexcoord;
uniform mat4 projection;
uniform mat4 view;

uniform float lightSpeed;
uniform vec3 cameraSpeed;

uniform sampler2D diffuseColorSampler;

// This only works for current scenes provided by the TAs 
// because the scenes we provide is transformed from gltf
uniform sampler2D normalMapSampler;
uniform sampler2D metallicRoughnessSampler;
bool isnan( float val )
{
  return ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
  // important: some nVidias failed to cope with version below.
  // Probably wrong optimization.
  /*return ( val <= 0.0 || 0.0 <= val ) ? false : true;*/
}

vec3 newColor(vec3 oldColor, float multiplier) {
	// return the color when wavelength multiplied by multiplier
    // TODO
    return oldColor;
}

void main() {
    position = vertexPosition;
    vec4 clipPos = projection * view * (vec4(position, 1.0));
    depth = clipPos.z / clipPos.w;
    texcoords = vTexcoord;

// Constants
vec3 beta = cameraSpeed / lightSpeed;
float gamma = 1 / sqrt(1 - dot(beta, beta));
float multiplier = 1;
if (length(beta) > 0.001)
{
// Get camera position
vec3 camPos = vec3(view[3]);
// Get the vector from the camera to the vertex
vec3 dir = position - camPos;
dir = normalize(dir);
// Calculate old direction
dir = normalize(dir + gamma * beta + (gamma - 1) * beta * dot(beta, dir) / dot(beta, beta));
// Get multiplier
multiplier = gamma * (1 - dot(beta, dir));
}

    diffuseColor = texture(diffuseColorSampler, vTexcoord).xyz;

    diffuseColor = newColor(diffuseColor, multiplier);
    
    metallicRoughness = texture(metallicRoughnessSampler, vTexcoord).yz;

    vec3 normalmap_value = texture(normalMapSampler, vTexcoord).xyz;
    vec3 norm = normalmap_value * 2.0 - 1.0;
    normal = vertexNormal;

    // HW6_TODO: Apply normal map here. Use normal textures to modify vertex normals.

    // Calculate tangent and bitangent
    vec3 edge1 = dFdx(vertexPosition);
    vec3 edge2 = dFdy(vertexPosition);
    vec2 deltaUV1 = dFdx(vTexcoord);
    vec2 deltaUV2 = dFdy(vTexcoord);

    vec3 tangent = edge1 * deltaUV2.y - edge2 * deltaUV1.y;

    // Robust tangent and bitangent evaluation
    if(length(tangent) < 1E-7) {
        vec3 bitangent = -edge1 * deltaUV2.x + edge2 * deltaUV1.x;
        if(length(bitangent) < 1E-7) {
            tangent = vec3(1, 0, 0);
            bitangent = vec3(0, 1, 0);
        }
        tangent = normalize(cross(bitangent, normal));
    }
    tangent = normalize(tangent - dot(tangent, normal) * normal);
    vec3 bitangent = normalize(cross(tangent,normal));

    mat3 TBN = mat3(tangent, bitangent, normal);

    vec3 normal_tmp = normalize(TBN * norm);
    normal = normal_tmp;
}