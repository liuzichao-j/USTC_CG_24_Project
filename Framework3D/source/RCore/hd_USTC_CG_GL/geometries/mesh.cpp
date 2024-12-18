//
// Copyright 2020 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//

#include "mesh.h"

#include <iostream>

#include "USTC_CG.h"
#include "Utils/Logging/Logging.h"
#include "context.h"
#include "../camera.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/imaging/hd/extComputationUtils.h"
#include "pxr/imaging/hd/instancer.h"
#include "pxr/imaging/hd/meshUtil.h"
#include "pxr/imaging/hd/smoothNormals.h"
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
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
#include "../renderParam.h"
#include "Nodes/GlobalUsdStage.h"
#include "Nodes/relativity/utils_relativity.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
using namespace pxr;
Hd_USTC_CG_Mesh::Hd_USTC_CG_Mesh(const SdfPath& id)
    : HdMesh(id),
      _cullStyle(HdCullStyleDontCare),
      _doubleSided(false),
      _normalsValid(false),
      _adjacencyValid(false),
      _refined(false)
{
}

Hd_USTC_CG_Mesh::~Hd_USTC_CG_Mesh()
{
}

HdDirtyBits Hd_USTC_CG_Mesh::GetInitialDirtyBitsMask() const
{
    int mask = HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints |
               HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyTransform |
               HdChangeTracker::DirtyVisibility | HdChangeTracker::DirtyCullStyle |
               HdChangeTracker::DirtyDoubleSided | HdChangeTracker::DirtyDisplayStyle |
               HdChangeTracker::DirtySubdivTags | HdChangeTracker::DirtyPrimvar |
               HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyInstancer |
               HdChangeTracker::DirtyMaterialId;

    return (HdDirtyBits)mask;
}

HdDirtyBits Hd_USTC_CG_Mesh::_PropagateDirtyBits(HdDirtyBits bits) const
{
    return bits;
}

TfTokenVector Hd_USTC_CG_Mesh::_UpdateComputedPrimvarSources(
    HdSceneDelegate* sceneDelegate,
    HdDirtyBits dirtyBits)
{
    HD_TRACE_FUNCTION();

    const SdfPath& id = GetId();

    // Get all the dirty computed primvars
    HdExtComputationPrimvarDescriptorVector dirtyCompPrimvars;
    for (size_t i = 0; i < HdInterpolationCount; ++i) {
        HdExtComputationPrimvarDescriptorVector compPrimvars;
        auto interp = static_cast<HdInterpolation>(i);
        compPrimvars = sceneDelegate->GetExtComputationPrimvarDescriptors(GetId(), interp);

        for (const auto& pv : compPrimvars) {
            if (HdChangeTracker::IsPrimvarDirty(dirtyBits, id, pv.name)) {
                dirtyCompPrimvars.emplace_back(pv);
            }
        }
    }

    if (dirtyCompPrimvars.empty()) {
        return TfTokenVector();
    }

    HdExtComputationUtils::ValueStore valueStore =
        HdExtComputationUtils::GetComputedPrimvarValues(dirtyCompPrimvars, sceneDelegate);

    TfTokenVector compPrimvarNames;
    // Update local primvar map and track the ones that were computed
    for (const auto& compPrimvar : dirtyCompPrimvars) {
        const auto it = valueStore.find(compPrimvar.name);
        if (!TF_VERIFY(it != valueStore.end())) {
            continue;
        }

        compPrimvarNames.emplace_back(compPrimvar.name);
        _primvarSourceMap[compPrimvar.name] = { it->second, compPrimvar.interpolation };
    }

    return compPrimvarNames;
}

void Hd_USTC_CG_Mesh::_UpdatePrimvarSources(HdSceneDelegate* sceneDelegate, HdDirtyBits dirtyBits)
{
    HD_TRACE_FUNCTION();
    const SdfPath& id = GetId();

    HdPrimvarDescriptorVector primvars;
    for (size_t i = 0; i < HdInterpolationCount; ++i) {
        auto interp = static_cast<HdInterpolation>(i);
        primvars = GetPrimvarDescriptors(sceneDelegate, interp);
        for (const HdPrimvarDescriptor& pv : primvars) {
            if (HdChangeTracker::IsPrimvarDirty(dirtyBits, id, pv.name) &&
                pv.name != HdTokens->points) {
                logging("Primvar source " + pv.name.GetString(), Info);
                _primvarSourceMap[pv.name] = { GetPrimvar(sceneDelegate, pv.name), interp };
            }
        }
    }
}

void Hd_USTC_CG_Mesh::_InitRepr(const TfToken& reprToken, HdDirtyBits* dirtyBits)
{
}

void Hd_USTC_CG_Mesh::_SetMaterialId(HdSceneDelegate* delegate, Hd_USTC_CG_Mesh* rprim)
{
    SdfPath const& newMaterialId = delegate->GetMaterialId(rprim->GetId());
    if (rprim->GetMaterialId() != newMaterialId) {
        rprim->SetMaterialId(newMaterialId);
    }
}

void Hd_USTC_CG_Mesh::Sync(
    HdSceneDelegate* sceneDelegate,
    HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits,
    const TfToken& reprToken)
{
    _dirtyBits = *dirtyBits;
    HD_TRACE_FUNCTION();
    HF_MALLOC_TAG_FUNCTION();

    _MeshReprConfig::DescArray descs = _GetReprDesc(reprToken);

    const SdfPath& id = GetId();
    std::string path = id.GetText();

    if (*dirtyBits & HdChangeTracker::DirtyMaterialId) {
        _SetMaterialId(sceneDelegate, this);
    }

    const auto& render_param = ((Hd_USTC_CG_RenderParam*)renderParam);
    auto limit_c_data = render_param->limited_light_speed_transform_data;
    auto stage = *render_param->global_usd_stage;

    bool able_relativity_light = limit_c_data->enable_limited_light_speed_transform;
    bool able_god_view = limit_c_data->enable_god_view;
    pxr::UsdGeomMesh usdgeom=pxr::UsdGeomMesh();
    
    // TODO: Check static objects.
    // Temporarily set geometry as dynamic objects
    if (path == "/geom/geometry")
        _static = false;
    
    if (able_relativity_light || able_god_view) 
    {
        if (stage) {
            for (auto stage_elm : stage->Traverse()) {
                TfToken typeId = stage_elm.GetTypeName();
                SdfPath sdf_path = stage_elm.GetPath();
                if (id == sdf_path) {
                    auto prim = stage->GetPrimAtPath(sdf_path);
                    usdgeom = pxr::UsdGeomMesh(prim);
                    break;
                }
            }
        }
    }

    if (able_god_view && usdgeom) {
        float time = *render_param->time_code;
        Hd_USTC_CG_Camera* free_camera;
        for (auto camera : *render_param->cameras) {
            if (camera->GetId() != SdfPath::EmptyPath()) {
                free_camera = camera;
                break;
            }
        }

        GfVec3f camera_position(free_camera->GetTransform().ExtractTranslation());
        
        float c = *render_param->speed_of_light;
        
        if (free_camera->_velocity.GetLength() >= c) {
            free_camera->_velocity.Normalize();
            free_camera->_velocity *= c;
        }

        pxr::GfVec3f beta = free_camera->_velocity / c;
        float gamma = 1 / sqrt(1 - beta.GetLengthSq());

        std::vector<double> time_samples;
        usdgeom.GetTimeSamples(&time_samples);

        std::vector<pxr::VtArray<pxr::GfVec3f>> hist_data_pos(time_samples.size());
        std::vector<pxr::GfMatrix4d> hist_data_transform(time_samples.size());
        // std::cout << time_samples.size() << std::endl;
        for (int i = 0; i < time_samples.size(); i++) {
            usdgeom.GetPointsAttr().Get(&hist_data_pos[i], time_samples[i]);
            hist_data_transform[i] = usdgeom.ComputeLocalToWorldTransform(time_samples[i]);
        }
        pxr::VtArray<pxr::GfVec3f> vertices, cur_velocity;
        pxr::GfMatrix4d vert_transform = usdgeom.ComputeLocalToWorldTransform(time);
        usdgeom.GetPointsAttr().Get(&vertices, time);
        cur_velocity.resize(vertices.size(), GfVec3f(0.0));

        int itr_n = limit_c_data->iteration_num;
        float damping = limit_c_data->iteration_damping;
        for (int i = 0; i < points.size(); i++) {
            float t = time;
            GfVec3f x, prev_x, next_x;
            double sample_dt, real_dt;
            for (int itr = 1; itr <= itr_n + 1; itr++) {
                // Newton Iteration
                int next_idx = std::lower_bound(time_samples.begin(), time_samples.end(), t) -
                               time_samples.begin();
                int prev_idx = next_idx - 1;
                if (next_idx >= time_samples.size() && time_samples.size() >= 2) {
                    prev_idx = time_samples.size() - 2;
                    next_idx = time_samples.size() - 1;
                }
                else if (
                    prev_idx < 0 || prev_idx >= time_samples.size() || next_idx < 0 ||
                    next_idx >= time_samples.size()) {
                    t = 0.0f;
                    break;
                }
                prev_x = hist_data_transform[prev_idx].TransformAffine(hist_data_pos[prev_idx][i]),
                next_x = hist_data_transform[next_idx].TransformAffine(hist_data_pos[next_idx][i]);
                sample_dt = time_samples[next_idx] - time_samples[prev_idx];
                real_dt = t - time_samples[prev_idx];
                double lambda = real_dt / sample_dt;
                x = prev_x * (1 - lambda) + next_x * lambda;

                if (itr == itr_n + 1)
                    break;
                double d = pxr::GfDot(x - camera_position, beta);
                double prev_d = pxr::GfDot(prev_x - camera_position, beta);

                double f = d - c * (t - time);
                double df = (d - prev_d) / real_dt - c;
                double step = 1;
                if (df != 0)
                    step = f / df * damping;
                t -= step;

                if (t < 0.0f) {
                    x = hist_data_transform[0].TransformAffine(hist_data_pos[0][i]);
                    d = pxr::GfDot(x - camera_position, beta);
                    t = d / c + time;
                    break;
                }
                if (t > time_samples.back()) {
                    x = hist_data_transform.back().TransformAffine((hist_data_pos.back())[i]);
                    d = pxr::GfDot(x - camera_position, beta);
                    t = d / c + time;
                    break;
                }
            }

            if (t <= 0.0f) {
                if (hist_data_transform.size() > 0) {
                    vertices[i] = hist_data_transform[0].TransformAffine(hist_data_pos[0][i]);
                    GfVec3f dir = vertices[i] - camera_position;
                    vertices[i] = vertices[i] +
                                  gamma * gamma / (gamma + 1) * pxr::GfDot(dir, beta) * beta -
                                  gamma * (t - time) * free_camera->_velocity;
                }
                cur_velocity[i] = GfVec3f(0.0);
            }
            else if (t >= time_samples.back()) {
                if (hist_data_transform.size() > 0) {
                    vertices[i] = hist_data_transform.back().TransformAffine((hist_data_pos.back())[i]);
                    GfVec3f dir = vertices[i] - camera_position;
                    vertices[i] = vertices[i] +
                                  gamma * gamma / (gamma + 1) * pxr::GfDot(dir, beta) * beta -
                                  gamma * (t - time) * free_camera->_velocity;
                }
                cur_velocity[i] = GfVec3f(0.0);
            }
            else {
                vertices[i] = x;
                GfVec3f dir = vertices[i] - camera_position;
                vertices[i] = vertices[i] +
                              gamma * gamma / (gamma + 1) * pxr::GfDot(dir, beta) * beta -
                              gamma * (t - time) * free_camera->_velocity;
                cur_velocity[i] = (x - prev_x) / real_dt;
            }
        }
        points = vertices;
        vertex_velocity = cur_velocity;
        transform.SetIdentity();
    }
    else if (!_static && able_relativity_light && usdgeom)
    {
		float time = *render_param->time_code;
		Hd_USTC_CG_Camera* free_camera;
		for (auto camera : *render_param->cameras) {
			if (camera->GetId() != SdfPath::EmptyPath()) {
				free_camera = camera;
				break;
			}
		}

		GfVec3d camera_position = free_camera->GetTransform().ExtractTranslation();

        float c = *render_param->speed_of_light;

        std::vector<double> time_samples;
        usdgeom.GetTimeSamples(&time_samples);

        std::vector<pxr::VtArray<pxr::GfVec3f>> hist_data_pos(time_samples.size());
        std::vector<pxr::GfMatrix4d> hist_data_transform(time_samples.size());
        for (int i = 0; i < time_samples.size(); i++)
        {
			usdgeom.GetPointsAttr().Get(&hist_data_pos[i], time_samples[i]); 
            hist_data_transform[i] = usdgeom.ComputeLocalToWorldTransform(time_samples[i]);
        }
        pxr::VtArray<pxr::GfVec3f> vertices, cur_velocity;
		pxr::GfMatrix4d vert_transform = usdgeom.ComputeLocalToWorldTransform(time);
		usdgeom.GetPointsAttr().Get(&vertices, time); 
        cur_velocity.resize(vertices.size(), GfVec3f(0.0));

        int itr_n = limit_c_data->iteration_num;
        float damping = limit_c_data->iteration_damping;
        for (int i = 0; i < points.size(); i++)
        {
			float t = time;
            GfVec3f x, v, prev_x, next_x;
			double sample_dt, real_dt;
            for (int itr = 1; itr <= itr_n + 1; itr++)
            {
                // Newton Iteration
				int next_idx = std::lower_bound(time_samples.begin(), time_samples.end(), t) - time_samples.begin();
				int prev_idx = next_idx - 1;
                if (next_idx >= time_samples.size() && time_samples.size() >= 2)
                {
                    prev_idx = time_samples.size() - 2;
					next_idx = time_samples.size() - 1;
				}
                else if (prev_idx < 0 || prev_idx >= time_samples.size() || next_idx < 0 || next_idx >= time_samples.size())
                {
                    t = 0.0f;
                    break;
                }
                prev_x = hist_data_transform[prev_idx].TransformAffine(hist_data_pos[prev_idx][i]),
				next_x = hist_data_transform[next_idx].TransformAffine(hist_data_pos[next_idx][i]);
                sample_dt = time_samples[next_idx] - time_samples[prev_idx];
                real_dt = t - time_samples[prev_idx];
                double lambda = real_dt / sample_dt;
                x = prev_x * (1 - lambda) + next_x * lambda;
                v = (next_x - prev_x) / sample_dt;

                if (itr == itr_n + 1)
                    break;
                double d = (x - camera_position).GetLength();
                
                double f = d + c * (t - time);
                double df = GfDot((x - camera_position).GetNormalized(), v) + c;
                double step = 1;
                if (df != 0) step = f / df * damping;
                t -= step;
                if (t < 0.0f) 
                {
					t = 0.0f;
					break;
                }
			}

            if (t == 0.0f)
            {
                if(hist_data_transform.size() > 0)
                    vertices[i] = hist_data_transform[0].TransformAffine(hist_data_pos[0][i]);
                cur_velocity[i] = GfVec3f(0.0);
            }
            else 
            {
                vertices[i] = x;
                cur_velocity[i] = (x - prev_x) / real_dt;
            }
		}
        points = vertices;
		vertex_velocity = cur_velocity;
		transform.SetIdentity();
    }
    else {
		if (HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points)) {
            VtValue value = sceneDelegate->Get(id, HdTokens->points);
            points = value.Get<VtVec3fArray>();

            _normalsValid = false;
		}
		if (HdChangeTracker::IsInstancerDirty(*dirtyBits, id) ||
			HdChangeTracker::IsTransformDirty(*dirtyBits, id)) {
			transform = GfMatrix4f(sceneDelegate->GetTransform(id));
		}
        pxr::VtArray<pxr::GfVec3f> cur_velocity;
        cur_velocity.resize(points.size(), GfVec3f(0.0));
		vertex_velocity = cur_velocity;
    }


    if (HdChangeTracker::IsTopologyDirty(*dirtyBits, id)) {
        topology = GetMeshTopology(sceneDelegate);
        HdMeshUtil meshUtil(&topology, GetId());
        meshUtil.ComputeTriangleIndices(&triangulatedIndices, &trianglePrimitiveParams);
        _normalsValid = false;
        _adjacencyValid = false;
    }

    if (HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->normals) ||
        HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->widths) ||
        HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->primvar)) {
        _UpdatePrimvarSources(sceneDelegate, *dirtyBits);
        _texcoordsClean = false;
    }

    if (!_adjacencyValid) {
        _adjacency.BuildAdjacencyTable(&topology);
        _adjacencyValid = true;
        // If we rebuilt the adjacency table, force a rebuild of normals.
        _normalsValid = false;
    }

    if (!_normalsValid) {
        computedNormals =
            Hd_SmoothNormals::ComputeSmoothNormals(&_adjacency, points.size(), points.cdata());
    }
    _UpdateComputedPrimvarSources(sceneDelegate, *dirtyBits);
    logging("Syncing mesh " + GetId().GetString(), Info);
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void Hd_USTC_CG_Mesh::RefreshGLBuffer()
{
    const SdfPath& id = GetId();

    if (HdChangeTracker::IsPrimvarDirty(_dirtyBits, id, HdTokens->points)) {
        // Generate and bind the VAO
        glDeleteVertexArrays(1, &VAO);
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        // Generate and bind the VBO
        glDeleteBuffers(1, &VBO);
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        // Upload the points data to the VBO
        glBufferData(
            GL_ARRAY_BUFFER, points.size() * sizeof(pxr::GfVec3f), points.cdata(), GL_STATIC_DRAW);

        // Specify the layout of the points in the VBO
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // Unbind the VAO and VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    if (HdChangeTracker::IsTopologyDirty(_dirtyBits, id)) {
        glBindVertexArray(VAO);

        glDeleteBuffers(1, &EBO);
        glGenBuffers(1, &EBO);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            triangulatedIndices.size() * sizeof(GfVec3i),
            triangulatedIndices.cdata(),
            GL_STATIC_DRAW);
    }
    if (!_normalsValid) {
        glBindVertexArray(VAO);

        glDeleteBuffers(1, &normalBuffer);
        glGenBuffers(1, &normalBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
        glBufferData(
            GL_ARRAY_BUFFER,
            computedNormals.size() * sizeof(GfVec3f),
            computedNormals.cdata(),
            GL_STATIC_DRAW);

        // Enable and specify the layout of the normal buffer
        glVertexAttribPointer(normalLocation, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
        glEnableVertexAttribArray(normalLocation);

        _normalsValid = true;
    }
    _dirtyBits = 0;
}

void Hd_USTC_CG_Mesh::RefreshTexcoordGLBuffer(TfToken texcoord_name)
{
    if (texcoord_name.IsEmpty()) {
        texcoord_name = texcoord_name = TfToken("UVMap");
    }
    if (!texcoord_name.IsEmpty()) {
        if (!_texcoordsClean) {
            if (!this->_primvarSourceMap[texcoord_name].data.IsEmpty()) {
                glDeleteBuffers(1, &texcoords);
                glGenBuffers(1, &texcoords);

                glBindBuffer(GL_SHADER_STORAGE_BUFFER, texcoords);

                logging(
                    GetId().GetString() +
                        " Attempts to attach texcoord: " + texcoord_name.GetString(),
                    Info);
                assert(this->_primvarSourceMap[texcoord_name].data.CanCast<VtVec2fArray>());

                VtArray<GfVec2f> raw_texcoord =
                    this->_primvarSourceMap[texcoord_name].data.Get<VtVec2fArray>();

                VtArray<GfVec2f> texcoord;

                if (this->_primvarSourceMap[texcoord_name].interpolation ==
                    HdInterpolationFaceVarying) {
                    HdMeshUtil mesh_util(&topology, GetId());

                    VtValue vt_triangulated;
                    mesh_util.ComputeTriangulatedFaceVaryingPrimvar(
                        raw_texcoord.cdata(),
                        raw_texcoord.size(),
                        HdTypeFloatVec2,
                        &vt_triangulated);
                    auto triangulated = vt_triangulated.Get<VtVec2fArray>();
                    texcoord.resize(points.size());
                    for (int i = 0; i < triangulatedIndices.size(); ++i) {
                        for (int j = 0; j < 3; ++j) {
                            texcoord[triangulatedIndices[i][j]] = triangulated[i * 3 + j];
                        }
                    }
                }
                else {
                    texcoord = raw_texcoord;
                }

                glBufferData(
                    GL_SHADER_STORAGE_BUFFER,
                    texcoord.size() * sizeof(GfVec2f),
                    texcoord.cdata(),
                    GL_STATIC_DRAW);
            }
            _texcoordsClean = true;
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, texcoords);
    }
}

void Hd_USTC_CG_Mesh::Finalize(HdRenderParam* renderParam)
{
    glDeleteVertexArrays(1, &VAO);
    VAO = 0;
    glDeleteBuffers(1, &VBO);
    VBO = 0;
    glDeleteBuffers(1, &EBO);
    EBO = 0;
    glDeleteBuffers(1, &normalBuffer);
    normalBuffer = 0;
    glDeleteBuffers(1, &texcoords);
    texcoords = 0;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
