layout(set = 0, binding = BINDING_SCENE) buffer SceneBufferInfo {
	Camera camera;
	int numSpheres;
	int numQuads;
	int numTriangles;

	int numEmitters;
};

struct BVHNode {
	vec4 aabbMinShapeIndex;
	vec4 aabbMaxExitIndex;
};

layout(set = 0, binding = BINDING_BVH) buffer BVH {
	BVHNode bvh[];
};

layout(set = 0, binding = BINDING_SPHERES) buffer Spheres {
	Sphere spheres[];
};

layout(set = 0, binding = BINDING_QUADS) buffer Quads {
	Quad quads[];
};



layout(set = 0, binding = BINDING_MATERIALS) buffer Materials {
	uint materials[];
};

struct Emitter {
	uint shape;
	float pdf;
	float cdf;
	float pad;
};

layout(set = 0, binding = BINDING_EMITTERS) buffer Emitters {
	Emitter emitters[];
};

void sampleShape(uint shape, out ShapeQueryRecord sRec) {
	if (shape < numSpheres) {
		sampleSphere(spheres[shape], sRec);
	} else if (shape < numSpheres+numQuads) {
		sampleQuad(quads[shape-numSpheres], sRec);
	} else {
		sampleTriangle(shape - numSpheres - numQuads, sRec);
	}
}

vec3 sampleEmitter(vec3 ref, out Ray shadowRay) {
	float emitterSample = randUniformFloat();
	int emitter = 0;
	// TODO: binary search
	for (int i = 0; i < numEmitters; i++) {
		emitterSample -= emitters[i].pdf;
		if (emitterSample < 0) {
			emitter = i;
			break;
		}
	}
	ShapeQueryRecord sRec;
	sampleShape(emitters[emitter].shape, sRec);

	uint mat = materials[emitters[emitter].shape];
	vec3 power = emissiveMaterials[mat & ((1<<MATERIAL_TAG_SHIFT)-1)].power;

	vec3 dir = sRec.p - ref;
	float dist = length(dir);
	dir /= dist;

	shadowRay.origin = ref;
	shadowRay.direction = dir;
	shadowRay.tMin = 2.*M_EPS;
	shadowRay.tMax = dist-M_EPS;

	float cosTheta = -dot(dir, sRec.n);
	if (cosTheta < 0) {
		return vec3(0.);
	}

	float pdf = emitters[emitter].pdf * sRec.pdf * dist*dist / cosTheta;

	return power / pdf;

}

bool intersectScene(Ray ray, out Intersection its);
bool intersectScene(Ray ray) {
	// TODO: optimize
	Intersection dummy;
	return intersectScene(ray, dummy);
}
bool intersectScene(Ray ray, out Intersection its) {
	its.objectID = -1;
#if USE_BVH == 1
	vec3 invRayDir = 1.0 / ray.direction;
	vec3 timeOffset = -ray.origin * invRayDir;
	for(uint currentNode = 0; currentNode < bvh.length(); ) {
		uint shapeIndex = floatBitsToUint(bvh[currentNode].aabbMinShapeIndex.w);
		uint exitIndex  = floatBitsToUint(bvh[currentNode].aabbMaxExitIndex.w);
		if (shapeIndex != -1) {
			bool hasIntersection;
			if (shapeIndex < numSpheres) {
				hasIntersection = intersectSphere(ray, spheres[shapeIndex], its);
			} else if (shapeIndex < numSpheres + numQuads) {
				hasIntersection = intersectQuad(ray, quads[shapeIndex-numSpheres], its);
			} else {
				uint ix = shapeIndex - numSpheres - numQuads;
				hasIntersection = intersectTriangle(ray, ix, its);
			}
			if (hasIntersection) {
				ray.tMax = its.t - M_EPS;
				its.objectID = int(shapeIndex);
			}
			currentNode = exitIndex;
		} else {
			vec3 tNegative = bvh[currentNode].aabbMinShapeIndex.xyz * invRayDir + timeOffset;
			vec3 tPositive = bvh[currentNode].aabbMaxExitIndex.xyz  * invRayDir + timeOffset;
			vec3 tMin = min(tNegative, tPositive);
			vec3 tMax = max(tNegative, tPositive);
			float t0 = max(max(tMin.x, tMin.y), tMin.z);
			float t1 = min(min(tMax.x, tMax.y), tMax.z);
			if (t0 < t1+M_EPS && t0 < ray.tMax && t1 > ray.tMin) {
				currentNode = currentNode+1; // continue in the tree
			} else {
				currentNode = exitIndex;
			}
		}
	}
#else
	if (numSpheres > 100 || numQuads > 100) {
		// failsafe
		return false;
	}

	for (int i = 0; i < numSpheres; i++) {
		if (intersectSphere(ray, spheres[i], its)) {
			ray.tMax = its.t - M_EPS;
			its.objectID = i;
		}
	}
	for (int i = 0; i < numQuads; i++) {
		if (intersectQuad(ray, quads[i], its)) {
			ray.tMax = its.t - M_EPS;
			its.objectID = numSpheres + i;
		}
	}
	for (int i = 0; i < numTriangles; i++) {
		if (intersectTriangle(ray, i, its)) {
			ray.tMax = its.t - M_EPS;
			its.objectID = numSpheres + numQuads + i;
		}
	}
#endif
	
	if (its.objectID == -1) {
		return false;
	}

	its.p = ray.origin + its.t * ray.direction;
	// TODO: generalize
	if (its.objectID < numSpheres) {
		populateSphereIntersection(spheres[its.objectID], its);
	} else if (its.objectID < numSpheres + numQuads) {
		populateQuadIntersection(quads[its.objectID-numSpheres], its);
	} else {
		populateTriangleIntersection(its.objectID-numSpheres-numQuads, its);
	}

	return true;
}
