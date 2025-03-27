/// Register all translations in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

namespace mlir {

namespace vitis {
void registerToVitisCppTranslation();
void registerToVitisTclTranslation();
} // namespace vitis
namespace dfg {
void registerToVivadoTclTranslation();
} // namespace dfg

inline void registerAllDFGMLIRTranslations()
{
    static bool initOnce = []() {
        dfg::registerToVivadoTclTranslation();
        vitis::registerToVitisCppTranslation();
        vitis::registerToVitisTclTranslation();
        return true;
    }();
    (void)initOnce;
}

} // namespace mlir
