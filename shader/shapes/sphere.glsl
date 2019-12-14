struct Sphere {
	vec4 positionRadius;
};

bool intersectSphere(Ray ray, Sphere sphere) {
	vec3 pos = sphere.positionRadius.xyz;
	float r  = sphere.positionRadius.w;

	vec3 l = ray.origin - pos;
	float b = 2 * dot(ray.direction, l);
	float c = dot(l, l) - r*r;
	float d = b*b-4*c;
	if (d < M_EPS) {
		return false;
	}
	return true;
}
bool intersectSphere(Ray ray, Sphere sphere, inout Intersection its) {
	vec3 pos = sphere.positionRadius.xyz;
	float r  = sphere.positionRadius.w;

	vec3 l = ray.origin - pos;
	float b = 2 * dot(ray.direction, l);
	float c = dot(l, l) - r*r;
	float d = b*b-4*c;
	if (d < 0) {
		return false;
	}
	d = sqrt(d);
	float t0 = -0.5*(b+d);
	if (ray.tMin <= t0 && t0 <= ray.tMax) {
		its.t = t0;
		return true;
	}
	float t1 = -0.5*(b-d);
	if (ray.tMin <= t1 && t1 <= ray.tMax) {
		its.t = t1;
		return true;
	}
	return false;
}

void populateSphereIntersection(Sphere sphere, inout Intersection its) {
	vec3 n = its.n = (its.p - sphere.positionRadius.xyz) / sphere.positionRadius.w;
	vec3 t = normalize(vec3(-n.z, 0., n.x));
	vec3 b = cross(n, t);
	its.frame = mat3(t, b, n);
	its.uv = vec2(0.5+atan(n.z,n.x)/(2*M_PI), 0.5+asin(clamp(n.y, -1, 1))/M_PI);
	if(isnan(its.uv.x)) {
		its.uv.x = 0.;
	}
}

void sampleSphere(Sphere sphere, out ShapeQueryRecord sRec) {
	sRec.n = randUniformSphere();
	sRec.p = sphere.positionRadius.xyz + sphere.positionRadius.w * sRec.n;
	sRec.pdf = 1. / (sphere.positionRadius.w * sphere.positionRadius.w * 4 * M_PI);
}

float pdfSampleSphere(Sphere sphere, ShapeQueryRecord sRec) {
	return 1. / (sphere.positionRadius.w * sphere.positionRadius.w * 4 * M_PI);
}

