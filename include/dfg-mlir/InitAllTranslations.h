/// Register all translations in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

namespace mlir {

namespace vitis {
void registerGenerateVitisProject();
} // namespace vitis
namespace dfg {
void registerToVivadoTclTranslation();
} // namespace dfg

inline void registerAllDFGMLIRTranslations()
{
    static bool initOnce = []() {
        vitis::registerGenerateVitisProject();
        return true;
    }();
    (void)initOnce;
}

} // namespace mlir
