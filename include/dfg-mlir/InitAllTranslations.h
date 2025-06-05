/// Register all translations in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

namespace mlir {

namespace emitHLS {
void registerGenerateemitHLSProject();
} // namespace emitHLS
namespace dfg {
void registerToVivadoTclTranslation();
} // namespace dfg

inline void registerAllDFGMLIRTranslations()
{
    static bool initOnce = []() {
        emitHLS::registerGenerateemitHLSProject();
        return true;
    }();
    (void)initOnce;
}

} // namespace mlir
