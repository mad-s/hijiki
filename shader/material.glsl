layout(set = 0, binding = BINDING_DIFFUSE) buffer DiffuseMaterials {
	 DiffuseMaterial diffuseMaterials[];
};

layout(set = 0, binding = BINDING_DIELECTRIC) buffer DielectricMaterials {
	 DielectricMaterial dielectricMaterials[];
};

layout(set = 0, binding = BINDING_EMISSIVE) buffer EmissiveMaterials {
	 EmissiveMaterial emissiveMaterials[];
};

layout(set = 0, binding = BINDING_TEXTURES) buffer TexturedMaterials {
	 TexturedMaterial texturedMaterials[];
};


layout(set = 0, binding = BINDING_PORTALS) buffer PortalMaterials {
	 PortalMaterial portalMaterials[];
};

layout(RGBA32F, set=0, binding = BINDING_TEXTURE_IMAGES) uniform image2DArray imgTextures;


vec3 evalBSDF(uint material, vec3 wi, Intersection its, vec3 wo) {
	uint tag = material >> MATERIAL_TAG_SHIFT;
	uint idx = material & ((1 << MATERIAL_TAG_SHIFT)-1);
	if (tag == MATERIAL_TAG_DIFFUSE) {
		vec3 color = diffuseMaterials[idx].color;
		return dot(its.n,wi) * color / M_PI;
	} else {
		return vec3(0.);
	}
}

// TODO: tangent space?
vec3 sampleBSDF(uint material, vec3 wi, inout Intersection its, out vec3 wo, inout vec3 extinction) {
	uint tag = material >> MATERIAL_TAG_SHIFT;
	uint idx = material & ((1 << MATERIAL_TAG_SHIFT)-1);
        vec3 wo_local;
	switch(tag) {
                case MATERIAL_TAG_PORTAL:
                        PortalMaterial material = portalMaterials[idx];
                        vec3 offset_from = its.p - material.origin_from;
                        vec4 out_offset = material.transform_to * material.transform_from * vec4(offset_from, 1.);
                        vec3 out_point = out_offset.xyz + material.origin_to;
                        its.p = out_point;
                        //wo = (material.transform_to * material.transform_from * vec4(wi, 1.)).xyz;
                        wo = wi;
                        return vec3(1.);
		case MATERIAL_TAG_DIFFUSE:
			wo_local = randCosHemisphere();
			//vec4 localToWorld = quaternionFromTo(vec3(0.,0.,1), n);
			//wo = quaternionRotate(wo_local, localToWorld);
			wo = its.frame * wo_local;
			return diffuseMaterials[idx].color;
                case MATERIAL_TAG_TEXTURE:
                        wo_local = randCosHemisphere();
                        wo = its.frame * wo_local;
                        //TODO: unhardcode height/width
                        uint textureIndex = texturedMaterials[idx].index;
                        return imageLoad(imgTextures, ivec3(its.uv * 512, textureIndex)).xyz;
		case MATERIAL_TAG_MIRROR:
			wo = reflect(wi, its.n);
			return vec3(1.);
		case MATERIAL_TAG_DIELECTRIC:
			float eta = dielectricMaterials[idx].extinction_etaRatio.w;
			float etaInv = 1. / eta;
			float cosThetaI = -dot(its.n, wi);
			vec3 normal = its.n;
                        bool isInsideDielectric = cosThetaI > 0;
			if (cosThetaI < 0) {
				eta = etaInv;
				etaInv = 1. / eta;
				normal = -normal;
				cosThetaI = -cosThetaI;
			}

			float k = 1.0 - etaInv*etaInv * (1-cosThetaI*cosThetaI);

			if (k <= 0) {
				// reflect
				wo = reflect(wi, normal);
			} else {
				float cosThetaO = sqrt(k);

				float rho_par  = (eta*cosThetaI-cosThetaO)/(eta*cosThetaI+cosThetaO);
				float rho_orth = (cosThetaI-eta*cosThetaO)/(cosThetaI+eta*cosThetaO);

				float f_r = 0.5 * (rho_par*rho_par + rho_orth*rho_orth);
				if (randUniformFloat() < f_r) {
					wo = reflect(wi, normal);
				} else {
                                        isInsideDielectric = !isInsideDielectric;
					vec3 parallel = wi - dot(wi, normal) * normal;
					// refract
					wo = etaInv * parallel - sqrt(k) * normal;
				}
			}
                        if (isInsideDielectric) {
                                extinction = dielectricMaterials[idx].extinction_etaRatio.rgb;
                        }
			return vec3(1.);
		case MATERIAL_TAG_EMISSIVE:
			return vec3(0.);

	}
}
