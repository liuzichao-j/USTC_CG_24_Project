#include "NODES_FILES_DIR.h"
#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "Nodes/socket_types/basic_socket_types.hpp"
#include "RCore/Backend.hpp"
#include "RCore/ResourceAllocator.hpp"
#include "camera.h"
#include "geometries/mesh.h"
#include "light.h"
#include "material.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hgiGL/computeCmds.h"
#include "render_node_base.h"
#include "resource_allocator_instance.hpp"
#include "rich_type_buffer.hpp"

#include "Nodes/GlobalUsdStage.h"
#include "Nodes/socket_types/geo_socket_types.hpp"
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdShade/material.h"
#include "pxr/usd/usdShade/materialBindingAPI.h"
#include "pxr/usd/usdSkel/animQuery.h"
#include "pxr/usd/usdSkel/cache.h"
#include "pxr/usd/usdSkel/animation.h"
#include "pxr/usd/usdSkel/bindingAPI.h"
#include "pxr/usd/usdSkel/skeletonQuery.h"
#include "GCore/Components/MaterialComponent.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/Components/SkelComponent.h"
#include "GCore/Components/XformComponent.h"
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdSkel/animQuery.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/skeleton.h>

#include <memory>

namespace USTC_CG::node_moving_relativity_rasterize {
bool legal(const std::string& string)
{
    if (string.empty()) {
        return false;
    }
    if (std::find_if(string.begin(), string.end(), [](char val) {
            return val == '(' || val == ')' || val == '-';
        }) == string.end()) {
        return true;
    }
    return false;
}

static void node_declare(NodeDeclarationBuilder& b)
{
    b.add_input<decl::Camera>("Camera");
    b.add_input<decl::Float>("Speed of Light").default_val(1.0f);
    b.add_input<decl::Float>("Time Code");
    b.add_output<decl::Meshes>("Meshes");
}

static void node_exec(ExeParams params)
{
    auto cameras = params.get_input<CameraArray>("Camera");
    auto speed_of_light = params.get_input<float>("Speed of Light");

    Hd_USTC_CG_Camera* free_camera;

    for (auto camera : cameras) {
        if (camera->GetId() != SdfPath::EmptyPath()) {
            free_camera = camera;
            break;
        }
    }

    auto size = free_camera->_dataWindow.GetSize();

    auto camera_mat = free_camera->GetTransform();
    GfVec3f camera_pos = { (float)camera_mat[3][0],
                           (float)camera_mat[3][1],
                           (float)camera_mat[3][2] };

    MeshArray meshes;

    float time_code = params.get_input<float>("Time Code");
    pxr::UsdTimeCode time = pxr::UsdTimeCode(time_code);
    if (time_code == 0) {
        time = pxr::UsdTimeCode::Default();
    }

    auto& stage = GlobalUsdStage::global_usd_stage;
    for (const auto& prim : stage->Traverse()) {
        if (prim.IsA<pxr::UsdGeomMesh>()) {
            pxr::UsdGeomMesh mesh(prim);
            if (!mesh) {
                continue;
            }
            pxr::UsdAttribute pointsAttr = mesh.GetPointsAttr();
            pxr::VtArray<pxr::GfVec3f> vertices;
            pointsAttr.Get(&vertices, pxr::UsdTimeCode(time));
            auto size = vertices.size();

            auto distance = [&](pxr::GfVec3f pos, float t) -> float {
                return -speed_of_light * speed_of_light * (t - time_code) * (t - time_code) +
                       (pos - camera_pos).GetLengthSq();
            };

            static float delta_time = 0.01;
            for (auto i = 0; i < size; ++i){
                auto min = 0.f, max = time_code;
                pxr::VtArray<pxr::GfVec3f> vertices_t;
                while (max - min > delta_time) {
                    auto mid = (min + max) / 2;
                    if (!pointsAttr.Get(&vertices_t, pxr::UsdTimeCode(mid)) || i >= vertices_t.size()) {
                        std::cout << "Error: Could not get the vertice" << i << "at time " << time
                                  << std::endl;
                        min = mid;
                        continue;
                    }
                    if (distance(vertices_t[i], mid) < 0)
                        min = mid;
                    else
                        max = mid;
                }
                vertices[i] = vertices_t[i];
            }

            // convert it to HdMesh
            // auto CG_Mesh = std::make_shared<Hd_USTC_CG_Mesh>(mesh.GetPath());
            // meshes.push_back(CG_Mesh.get());

            auto CG_Mesh = new Hd_USTC_CG_Mesh(mesh.GetPath());
            meshes.push_back(CG_Mesh);
        }
    }
    params.set_output("Meshes", meshes);
}

static void node_register()
{
    static NodeTypeInfo ntype;

    strcpy(ntype.ui_name, "Moving Meshes");
    strcpy_s(ntype.id_name, "render_moving_meshes");

    render_node_type_base(&ntype);
    ntype.node_execute = node_exec;
    ntype.declare = node_declare;

    nodeRegisterType(&ntype);
}

NOD_REGISTER_NODE(node_register)
}  // namespace USTC_CG::node_moving_meshes