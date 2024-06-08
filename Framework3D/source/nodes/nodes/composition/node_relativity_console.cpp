#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "comp_node_base.h"
#include "Nodes/GlobalUsdStage.h"

namespace USTC_CG::node_relativity_console {

static void node_declare(NodeDeclarationBuilder& b)
{
    GlobalUsdStage::relativity_console_bind_data.clear();

    b.add_input<decl::Float>("Speed of Light").min(0.0).max(100.0).default_val(50.0);
    GlobalUsdStage::relativity_console_bind_data.push_back(&GlobalUsdStage::speed_of_light);

    b.add_input<decl::Float>("Max Camera Speed").min(0.0).max(1.0).default_val(0.9);
    GlobalUsdStage::relativity_console_bind_data.push_back(&GlobalUsdStage::camera_max_speed);

    b.add_input<decl::Int>("Use Limited-C Transform").min(0).max(1).default_val(1);
    GlobalUsdStage::relativity_console_bind_data.push_back(&GlobalUsdStage::enable_limited_light_speed_transform);

    b.add_input<decl::Int>("Newton Iteration Number").min(1).max(10).default_val(4);
    GlobalUsdStage::relativity_console_bind_data.push_back(&GlobalUsdStage::iteration_num);

    b.add_input<decl::Float>("Iteration Damping").min(0).max(1).default_val(0.5);
    GlobalUsdStage::relativity_console_bind_data.push_back(&GlobalUsdStage::iteration_damping);

    b.add_input<decl::Int>("Use God View").min(0).max(1).default_val(0);
    GlobalUsdStage::relativity_console_bind_data.push_back(&GlobalUsdStage::enable_god_view);

}

static void node_exec(ExeParams params)
{
    // This is for external read. Do nothing.
}

static void node_register()
{
    static NodeTypeInfo ntype;
    strcpy(ntype.ui_name, "Relativity Console");
    strcpy_s(ntype.id_name, "comp_relativity_console");
    comp_node_type_base(&ntype);
    ntype.node_execute = node_exec;
    ntype.declare = node_declare;
    ntype.ALWAYS_REQUIRED = true;
    nodeRegisterType(&ntype);
}

NOD_REGISTER_NODE(node_register)
}  // namespace USTC_CG::node_relativity_console