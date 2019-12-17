#define M_PI 3.1415926535897932384626433832795
#define M_EPS 1e-4

vec2 toSphericalCoords(vec3 direction) {
        float theta = acos(direction.y);
        float phi = atan(direction.z, direction.x) + M_PI;
        return vec2(phi, theta);
}

vec2 sphericalToUvCoords(vec2 spherical) {
        return vec2(spherical.x / 2. / M_PI, spherical.y / M_PI);
}
