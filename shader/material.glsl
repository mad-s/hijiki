layout(set = 0, binding = 7) buffer DiffuseMaterials {
	 DiffuseMaterial diffuseMaterials[];
};

layout(set = 0, binding = 8) buffer DielectricMaterials {
	 DielectricMaterial dielectricMaterials[];
};

layout(set = 0, binding = 9) buffer EmissiveMaterials {
	 EmissiveMaterial emissiveMaterials[];
};

/*
layout(set = 0, binding = 10) buffer PortalMaterials {
	 PortalMaterial portalMaterials[];
};
*/


vec3 evalBSDF(uint material, vec3 wi, vec3 n, vec3 wo) {
	uint tag = material >> MATERIAL_TAG_SHIFT;
	uint idx = material & ((1 << MATERIAL_TAG_SHIFT)-1);
	if (tag == MATERIAL_TAG_DIFFUSE) {
		return dot(n,wi) * diffuseMaterials[idx].color / M_PI;
	} else {
		return vec3(0.);
	}
}

// TODO: tangent space?
vec3 sampleBSDF(uint material, vec3 wi, vec3 n, out vec3 wo, inout vec3 extinction) {
	uint tag = material >> MATERIAL_TAG_SHIFT;
	uint idx = material & ((1 << MATERIAL_TAG_SHIFT)-1);
	switch(tag) {
		case MATERIAL_TAG_DIFFUSE:
			vec3 wo_local = randCosHemisphere();
			vec4 localToWorld = quaternionFromTo(vec3(0.,0.,1), n);
			wo = quaternionRotate(wo_local, localToWorld);
			return diffuseMaterials[idx].color;
		case MATERIAL_TAG_MIRROR:
			wo = reflect(wi, n);
			return vec3(1.);
		case MATERIAL_TAG_DIELECTRIC:
			float eta = dielectricMaterials[idx].extinction_etaRatio.w;
			float etaInv = 1. / eta;
			float cosThetaI = -dot(n, wi);
			vec3 normal = n;
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
