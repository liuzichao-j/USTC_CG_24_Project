#version 430

layout(location = 0) out vec3 position;
layout(location = 1) out float depth;
layout(location = 2) out vec2 texcoords;
layout(location = 3) out vec3 diffuseColor;
layout(location = 4) out vec2 metallicRoughness;
layout(location = 5) out vec3 normal;

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec3 vertexVelocity;
in vec2 vTexcoord;
uniform mat4 projection;
uniform mat4 view;

uniform float lightSpeed;
uniform float dopplerMix;
uniform vec3 camPos;
uniform vec3 camSpeed;

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

vec3 col_palette[9] = {
    vec3(0.4, 0.0, 0.5),    // UV
    vec3(0.8, 0.0, 1.0),    // Viovet
    vec3(0.0, 0.0, 1.0),    // Blue
    vec3(0.0, 1.0, 1.0),    // Cyan
    vec3(0.0, 1.0, 0.0),    // Green
    vec3(1.0, 1.0, 0.0),    // Yellow
    vec3(1.0, 0.0, 0.0),    // Orange
    vec3(1.0, 0.0, 0.0),    // Red
    vec3(0.5, 0.0, 0.0)     // Infrared
};

float ratio(float x, float l, float r)
{
    return (x - l) / (r - l);
}

vec3 wavelengthToRGB(float wavelength)
{
    float r, g, b, factor;
    vec3 col = vec3(0.0);
    if(wavelength < 380)
        col = mix(col_palette[0], col_palette[1], pow(2, (wavelength - 380) / 100));
    else if (wavelength >= 380 && wavelength < 440) 
        col = mix(col_palette[1], col_palette[2], ratio(wavelength, 380, 440));
    else if (wavelength >= 440 && wavelength < 490) 
        col = mix(col_palette[2], col_palette[3], ratio(wavelength, 440, 490));
    else if (wavelength >= 490 && wavelength < 510) 
        col = mix(col_palette[3], col_palette[4], ratio(wavelength, 490, 510));
    else if (wavelength >= 510 && wavelength < 580) 
        col = mix(col_palette[4], col_palette[5], ratio(wavelength, 510, 580));
    else if (wavelength >= 580 && wavelength < 645) 
        col = mix(col_palette[5], col_palette[6], ratio(wavelength, 580, 645));
    else if (wavelength >= 645 && wavelength < 780) 
        col = mix(col_palette[6], col_palette[7], ratio(wavelength, 645, 780));
    else if(wavelength > 780)
        col = mix(col_palette[7], col_palette[8], pow(2, (780 - wavelength) / 100));
    
    if (wavelength >= 380 && wavelength < 420)
        factor = mix(0.5, 1.0, ratio(wavelength, 380, 420));
    else if (wavelength >= 420 && wavelength < 700)
    	factor = 1.0;
    else if (wavelength >= 700 && wavelength < 780)
        factor = mix(1.0, 0.5, ratio(wavelength, 700, 780));
    else
    	factor = 0.5;

    if(factor != 0) 
    {
        col.r = pow(col.r * factor, 0.8);
        col.g = pow(col.g * factor, 0.8);
        col.b = pow(col.b * factor, 0.8);
    }
    return col;

}

vec3 newColor(vec3 oldColor, float multiplier) {
	// return the color when wavelength multiplied by multiplier
    // Use simple method. Seperates two wavelengths and tackle them seperately.
    float wavelength1, wavelength2, intensity1, intensity2;
    // First: seperate oldColor
    float r = pow(oldColor.r, 1.25), g = pow(oldColor.g, 1.25), b = pow(oldColor.b, 1.25);
    float max = max(r, max(g, b)), min = min(r, min(g, b));
    if (r == max) {
        // Try bigger wavelength
        if (b == min) {
            wavelength1 = 490, intensity1 = b;
            wavelength2 = 645 - 65 * (g - b) / r, intensity2 = r;
        }
        else {
            wavelength1 = 645, intensity1 = r;
            wavelength2 = 440 + 50 * g / b, intensity2 = b;
        }
    }
    else if (g == max) {
        // Try medium wavelength
        wavelength1 = 645, intensity1 = r;
        wavelength2 = 510 - 20 * b / g, intensity2 = g;
    }
    else {
		// Try smaller wavelength
        wavelength1 = 440 - 60 * r / b, intensity1 = b;
        if (wavelength1 >= 380 && wavelength1 < 420) {
            intensity1 /= (0.3 + 0.7 * (wavelength1 - 380) / 40);
        }
        wavelength2 = 510, intensity2 = g;
	}
    // Second: calculate newColor
    wavelength1 *= multiplier, wavelength2 *= multiplier;
    vec3 newColor = intensity1 * wavelengthToRGB(wavelength1) + intensity2 * wavelengthToRGB(wavelength2);
    return newColor;
}

void main() {
    position = vertexPosition;
    vec4 clipPos = projection * view * (vec4(position, 1.0));
    depth = clipPos.z / clipPos.w;
    texcoords = vTexcoord;

    // Constants
    vec3 beta = (camSpeed - vertexVelocity) / lightSpeed;
    float gamma = 1 / sqrt(1 - dot(beta, beta));
    float multiplier = 1;
    if (length(beta) > 0)
    {
        // Get the vector from the camera to the vertex
        vec3 dir = position - camPos;
        dir = normalize(dir);
        // Calculate old direction
        dir = normalize(dir + gamma * beta + (gamma - 1) * beta * dot(beta, dir) / dot(beta, beta));
        // Get multiplier
        multiplier = gamma * (1 - dot(beta, dir));
    }

    vec3 textureColor = texture(diffuseColorSampler, vTexcoord).xyz;
    vec3 dopplerColor;
    if(isnan(multiplier))
        dopplerColor = vec3(1.0, 0.0, 0.0);
    else
        dopplerColor = newColor(textureColor, multiplier);

    diffuseColor = mix(textureColor, dopplerColor, dopplerMix);


    metallicRoughness = texture(metallicRoughnessSampler, vTexcoord).yz;

    vec3 normalmap_value = texture(normalMapSampler, vTexcoord).xyz;
    vec3 norm = normalmap_value * 2.0 - 1.0;
    normal = vertexNormal;

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