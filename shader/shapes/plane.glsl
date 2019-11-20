struct Plane {
	vec4 normalOffset;
};

bool intersectPlane(Ray ray, Plane plane) {
	vec3 n  = plane.normalOffset.xyz;
	if (dot(ray.direction, n) > 0) {
		return false;
	}
	float a = plane.normalOffset.w;
	float t = -(dot(ray.origin, n)+a) / dot(ray.direction, n);
	if (ray.tMin <= t && t <= ray.tMax) {
		return true;
	}
	return false;
}
bool intersectPlane(Ray ray, Plane plane, inout Intersection its) {
	vec3 n  = plane.normalOffset.xyz;
	if (dot(ray.direction, n) > 0) {
		return false;
	}
	float a = plane.normalOffset.w;
	float t = -(dot(ray.origin, n)+a) / dot(ray.direction, n);
	if (ray.tMin <= t && t <= ray.tMax) {
		its.t = t;
		return true;
	}
	return false;
}
