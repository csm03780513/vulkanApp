

#include <filesystem>
#include "HelloTriangleApplication.cpp"

int main() {
    HelloTriangleApplication app;

    try {
        std::cout << "Current working directory: "
                  << std::filesystem::current_path() << std::endl;
        app.run();

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
