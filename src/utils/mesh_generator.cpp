#include "utils/mesh_generator.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace mesh_generator {
    bool is_inline_geo(const std::string &geo_spec) {
        // Check if geo_spec starts with the inline_geo_prefix
        return geo_spec.rfind(inline_geo_prefix, 0) == 0;
    }

    std::string gmsh_generate_msh(const std::string &geo_spec, const std::string &msh_path) {
        // Validate output .msh path
        const std::filesystem::path msh_out(msh_path);
        if (!msh_out.has_extension() || msh_out.extension() != ".msh") {
            throw std::runtime_error(
                    "mesh_generator::gmsh_generate_msh: output path must end with .msh: " +
                    msh_path);
        }

        // geo_path will hold either the path to the .geo file, or the path to a temporary .geo file
        std::filesystem::path geo_path;
        // tmp_geo_path holds the path to a temporary .geo file if we create one
        std::filesystem::path tmp_geo_path;

        // Handle inline geo script
        if (is_inline_geo(geo_spec)) {
            // Extract the geo script contents
            std::string geo_contents = geo_spec.substr(std::strlen(inline_geo_prefix));
            // Allow an optional leading newline after the prefix
            if (!geo_contents.empty() && geo_contents.front() == '\n')
                // Remove leading newline
                geo_contents.erase(geo_contents.begin());

            // Create output directory if it doesn't exist
            std::error_code ec;
            std::filesystem::create_directories(msh_out.parent_path(), ec);
            if (ec) {
                throw std::runtime_error(
                        "mesh_generator::gmsh_generate_msh: failed to create output directory: " +
                        msh_out.parent_path().string());
            }

            // Write the geo_contents to a temporary .geo file
            tmp_geo_path = msh_out;
            tmp_geo_path.replace_extension(".geo");

            std::ofstream out(tmp_geo_path);
            if (!out.good()) {
                throw std::runtime_error(
                        "mesh_generator::gmsh_generate_msh: cannot write temporary .geo file: " +
                        tmp_geo_path.string());
            }
            out << geo_contents;
            geo_path = tmp_geo_path;
        } else {
            // Treat geo_spec as a .geo file path
            geo_path = std::filesystem::path(geo_spec);
            if (!geo_path.has_extension() || geo_path.extension() != ".geo") {
                throw std::runtime_error("mesh_generator::gmsh_generate_msh expects a .geo path or "
                                         "an inline_geo: script.");
            }
        }

        // Construct and execute the gmsh command
        const std::string cmd = "gmsh -2 "
                                "-format msh2 "
                                "-setnumber Mesh.MshFileVersion 2.2 "
                                "-o \"" +
                                msh_out.string() + "\" \"" + geo_path.string() + "\"";

        // Execute the command
        if (const int ret = std::system(cmd.c_str()); ret != 0) {
            // Clean up temporary .geo file if created
            if (!tmp_geo_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(tmp_geo_path, ec);
            }
            throw std::runtime_error("mesh_generator::gmsh_generate_msh: gmsh failed. Command: " +
                                     cmd);
        }

        // Clean up temporary .geo file if created
        if (!tmp_geo_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(tmp_geo_path, ec);
        }

        // Return the output .msh path
        return msh_out.string();
    }

} // namespace mesh_generator
