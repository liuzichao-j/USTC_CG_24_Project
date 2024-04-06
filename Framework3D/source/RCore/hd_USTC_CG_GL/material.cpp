// #define __GNUC__

#include "material.h"

#include "Utils/Logging/Logging.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/imaging/hio/image.h"
#include "pxr/usd/sdr/shaderNode.h"
#include "pxr/usd/usd/tokens.h"
#include "pxr/usdImaging/usdImaging/tokens.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
using namespace pxr;

// Here for the cource purpose, we support a very limited set of forms of the material.
// Specifically, we support only UsdPreviewSurface, and each input can be either value, or a texture
// connected to a primvar reader.

HdMaterialNode2 Hd_USTC_CG_Material::get_input_connection(
    HdMaterialNetwork2 surfaceNetwork,
    std::map<TfToken, std::vector<HdMaterialConnection2>>::value_type& input_connection)
{
    HdMaterialNode2 upstream;
    assert(input_connection.second.size() == 1);
    upstream = surfaceNetwork.nodes[input_connection.second[0].upstreamNode];
    return upstream;
}

void Hd_USTC_CG_Material::DestroyTexture(InputDescriptor& input_descriptor)
{
    if (input_descriptor.glTexture) {
        glDeleteTextures(1, &input_descriptor.glTexture);
        input_descriptor.glTexture = 0;
    }
}

void Hd_USTC_CG_Material::TryLoadTexture(
    const char* name,
    InputDescriptor& descriptor,
    HdMaterialNode2& usd_preview_surface)
{
    for (auto&& input_connection : usd_preview_surface.inputConnections) {
        if (input_connection.first == TfToken(name)) {
            logging("Loading texture: " + input_connection.first.GetString());
            auto texture_node = get_input_connection(surfaceNetwork, input_connection);
            assert(texture_node.nodeTypeId == UsdImagingTokens->UsdUVTexture);

            auto file_name =
                texture_node.parameters[TfToken("file")].Get<SdfAssetPath>().GetAssetPath();
            logging("Texture file name: " + file_name);

            descriptor.image = HioImage::OpenForReading(file_name);
            descriptor.wrapS = texture_node.parameters[TfToken("wrapS")].Get<TfToken>();
            descriptor.wrapT = texture_node.parameters[TfToken("wrapT")].Get<TfToken>();

            HdMaterialNode2 st_read_node;
            for (auto&& st_read_connection : texture_node.inputConnections) {
                st_read_node = get_input_connection(surfaceNetwork, st_read_connection);
            }

            assert(st_read_node.nodeTypeId == UsdImagingTokens->UsdPrimvarReader_float2);
            descriptor.uv_primvar_name = st_read_node.parameters[TfToken("varname")].Get<TfToken>();
        }
    }
}

void Hd_USTC_CG_Material::TryLoadParameter(
    const char* name,
    InputDescriptor& descriptor,
    HdMaterialNode2& usd_preview_surface)
{
    for (auto&& parameter : usd_preview_surface.parameters) {
        if (parameter.first == name) {
            descriptor.value = parameter.second;
            logging("Loading parameter: " + parameter.first.GetString());
        }
    }
}

static HdFormat hdFormatConversion(HioFormat format)
{
    switch (format) {
        case HioFormatUNorm8: return HdFormatUNorm8;
        case HioFormatUNorm8Vec2: return HdFormatUNorm8Vec2;
        case HioFormatUNorm8Vec3: return HdFormatUNorm8Vec3;
        case HioFormatUNorm8Vec4: return HdFormatUNorm8Vec4;
        case HioFormatSNorm8: return HdFormatSNorm8;
        case HioFormatSNorm8Vec2: return HdFormatSNorm8Vec2;
        case HioFormatSNorm8Vec3: return HdFormatSNorm8Vec3;
        case HioFormatSNorm8Vec4: return HdFormatSNorm8Vec4;
        case HioFormatFloat16: return HdFormatFloat16;
        case HioFormatFloat16Vec2: return HdFormatFloat16Vec2;
        case HioFormatFloat16Vec3: return HdFormatFloat16Vec3;
        case HioFormatFloat16Vec4: return HdFormatFloat16Vec4;
        case HioFormatFloat32: return HdFormatFloat32;
        case HioFormatFloat32Vec2: return HdFormatFloat32Vec2;
        case HioFormatFloat32Vec3: return HdFormatFloat32Vec3;
        case HioFormatFloat32Vec4: return HdFormatFloat32Vec4;
        case HioFormatInt16: return HdFormatInt16;
        case HioFormatInt16Vec2: return HdFormatInt16Vec2;
        case HioFormatInt16Vec3: return HdFormatInt16Vec3;
        case HioFormatInt16Vec4: return HdFormatInt16Vec4;
        case HioFormatUInt16: return HdFormatUInt16;
        case HioFormatUInt16Vec2: return HdFormatUInt16Vec2;
        case HioFormatUInt16Vec3: return HdFormatUInt16Vec3;
        case HioFormatUInt16Vec4: return HdFormatUInt16Vec4;
        case HioFormatInt32: return HdFormatInt32;
        case HioFormatInt32Vec2: return HdFormatInt32Vec2;
        case HioFormatInt32Vec3: return HdFormatInt32Vec3;
        case HioFormatInt32Vec4: return HdFormatInt32Vec4;
        default: return HdFormatInvalid;
    }
}

// Function to create an OpenGL texture from a HioImage object
static GLuint createTextureFromHioImage(const HioImageSharedPtr& image)
{
    // Step 1: Get image information
    int width = image->GetWidth();
    int height = image->GetHeight();
    HioFormat format = image->GetFormat();

    // Step 2: Allocate memory for storing image data
    HioImage::StorageSpec storageSpec;
    storageSpec.width = width;
    storageSpec.height = height;
    storageSpec.format = format;
    storageSpec.data = malloc(width * height * image->GetBytesPerPixel());
    if (!storageSpec.data) {
        // Handle error if unable to allocate memory
        return 0;
    }

    // Step 3: Read the image data
    if (!image->Read(storageSpec)) {
        // Handle error if unable to read image data
        free(storageSpec.data);
        return 0;
    }

    // Step 4: Create an OpenGL texture object
    GLuint texture;
    glGenTextures(1, &texture);

    // Step 5: Bind the texture object and specify its parameters
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, storageSpec.data);

    // Step 6: Optionally set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Step 7: Unbind the texture object
    glBindTexture(GL_TEXTURE_2D, 0);

    // Step 8: Clean up allocated memory
    free(storageSpec.data);

    return texture;
}

void Hd_USTC_CG_Material::TryCreateGLTexture(InputDescriptor& descriptor)
{
    if (descriptor.image) {
        if (descriptor.glTexture == 0) {
            descriptor.glTexture = createTextureFromHioImage(descriptor.image);
        }
    }
}

#define INPUT_LIST                                                                       \
    diffuseColor, specularColor, emissiveColor, displacement, opacity, opacityThreshold, \
        roughness, metallic, clearcoat, clearcoatRoughness, occlusion, normal, ior

#define TRY_LOAD(INPUT)                                 \
    TryLoadTexture(#INPUT, INPUT, usd_preview_surface); \
    TryLoadParameter(#INPUT, INPUT, usd_preview_surface);

void Hd_USTC_CG_Material::Sync(
    HdSceneDelegate* sceneDelegate,
    HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    VtValue vtMat = sceneDelegate->GetMaterialResource(GetId());
    if (vtMat.IsHolding<HdMaterialNetworkMap>()) {
        HdMaterialNetworkMap const& hdNetworkMap = vtMat.UncheckedGet<HdMaterialNetworkMap>();
        if (!hdNetworkMap.terminals.empty() && !hdNetworkMap.map.empty()) {
            logging("Loaded a material", Info);

            surfaceNetwork = HdConvertToHdMaterialNetwork2(hdNetworkMap);

            // Here we only support single output material.
            assert(surfaceNetwork.terminals.size() == 1);

            auto terminal = surfaceNetwork.terminals[HdMaterialTerminalTokens->surface];

            auto usd_preview_surface = surfaceNetwork.nodes[terminal.upstreamNode];
            assert(usd_preview_surface.nodeTypeId == UsdImagingTokens->UsdPreviewSurface);

            MACRO_MAP(TRY_LOAD, INPUT_LIST)
        }
    }
    else {
        logging("Not loaded a material", Info);
    }
    *dirtyBits = Clean;
}

void Hd_USTC_CG_Material::RefreshGLBuffer()
{
    MACRO_MAP(; TryCreateGLTexture, INPUT_LIST);
}

HdDirtyBits Hd_USTC_CG_Material::GetInitialDirtyBitsMask() const
{
    return AllDirty;
}

void Hd_USTC_CG_Material::Finalize(HdRenderParam* renderParam)
{
    MACRO_MAP(; DestroyTexture, INPUT_LIST);

    HdMaterial::Finalize(renderParam);
}

USTC_CG_NAMESPACE_CLOSE_SCOPE