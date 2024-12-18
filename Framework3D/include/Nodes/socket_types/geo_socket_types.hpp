#pragma once

#include "USTC_CG.h"
#include "basic_socket_types.hpp"
#include "buffer_socket_types.hpp"
#include "Nodes/make_standard_type.hpp"
#include "Nodes/node_declare.hpp"
USTC_CG_NAMESPACE_OPEN_SCOPE
namespace decl {

DECLARE_SOCKET_TYPE(Geometry)
DECLARE_SOCKET_TYPE(MassSpringSocket)
DECLARE_SOCKET_TYPE(RelativityMassSpringSocket)
DECLARE_SOCKET_TYPE(SPHFluidSocket)
DECLARE_SOCKET_TYPE(AnimatorSocket)

}  // namespace decl

USTC_CG_NAMESPACE_CLOSE_SCOPE