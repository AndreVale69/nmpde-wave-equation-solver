#ifndef MESH_GENERATOR_HPP
#define MESH_GENERATOR_HPP

#include <filesystem>
#include <string>

/**
 * Small helper module to generate .msh meshes using gmsh.
 *
 * It supports two input modes:
 *  - a path to a .geo file
 *  - an inline .geo script, prefixed with "inline_geo:"
 *
 * The module is intentionally tiny and independent of deal.II and MPI.
 */
namespace mesh_generator {
    /**
     * @brief Prefix to indicate an inline .geo script. If a geo_spec starts with this prefix,
     *        the rest of the string is treated as the full .geo script.
     */
    inline constexpr auto inline_geo_prefix = "inline_geo:";

    /**
     * @brief Check whether the given geo_spec is an inline geo script.
     *
     * @param geo_spec  Either a filesystem path ending in .geo, or a string starting with
     *                  mesh_generator::inline_geo_prefix followed by the full .geo script.
     * @return          true if geo_spec is an inline geo script, false if it is a .geo file path.
     */
    bool is_inline_geo(const std::string &geo_spec);

    /**
     * @brief Generate a .msh from either a .geo path or an inline geo script.
     *
     * @param geo_spec  Either a filesystem path ending in .geo, or a string starting with
     *                  mesh_generator::inline_geo_prefix followed by the full .geo script.
     * @param msh_path  Output mesh file path; must end with .msh.
     *
     * @throws via AssertThrow in callers (this function itself throws std::runtime_error).
     */
    std::string gmsh_generate_msh(const std::string &geo_spec, const std::string &msh_path);

} // namespace mesh_generator

#endif // MESH_GENERATOR_HPP
