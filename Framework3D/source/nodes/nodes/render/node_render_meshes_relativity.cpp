#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "render_node_base.h"
#include "Nodes/socket_types/basic_socket_types.hpp"
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

#include "GCore/Components/MaterialComponent.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/Components/SkelComponent.h"
#include "GCore/Components/XformComponent.h"
#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "pxr/usd/usdSkel/animation.h"
#include "pxr/usd/usdSkel/bindingAPI.h"
#include "pxr/usd/usdSkel/skeletonQuery.h"

namespace USTC_CG::node_scene_meshes_relativity {
static void node_declare(NodeDeclarationBuilder& b)
{
    b.add_input<decl::Usd>("Global USD Stage");
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
    float time_code = params.get_input<float>("Time Code");
    float time = time_code;

    GOperandBase geometry;
    std::shared_ptr<MeshComponent> mesh = std::make_shared<MeshComponent>(&geometry);
    geometry.attach_component(mesh);
    pxr::UsdStageWeakPtr* stage_ptr = params.get_input<pxr::UsdStageWeakPtr*>("Global USD Stage");
    pxr::UsdStageWeakPtr stage = *stage_ptr;

    if (stage) {
        for (auto stage_elm : stage->Traverse())
        {
            TfToken typeId = stage_elm.GetTypeName();
            SdfPath sdf_path = stage_elm.GetPath();
			std::cout << "reading: " << sdf_path.GetText() << std::endl;
			if (typeId == HdPrimTypeTokens->mesh) {
				auto prim = stage->GetPrimAtPath(sdf_path);
				pxr::UsdGeomMesh usdgeom(prim);


				// VBO, copy to gpu,...

			}
        }
		throw std::runtime_error("Not implemented.");
    }
    else {
		throw std::runtime_error("Unable to read Global USD Stage.");
    }
    params.set_output("Meshes", std::move(geometry));
}

static void node_register()
{
    static NodeTypeInfo ntype;

    strcpy(ntype.ui_name, "Scene Meshes (Relativity)");
    strcpy_s(ntype.id_name, "render_scene_meshes_relativity");

    render_node_type_base(&ntype);
    ntype.node_execute = node_exec;
    ntype.declare = node_declare;
    nodeRegisterType(&ntype);
}

NOD_REGISTER_NODE(node_register)
}  // namespace USTC_CG::node_scene_meshes
