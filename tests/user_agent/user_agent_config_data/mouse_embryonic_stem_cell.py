from ..models import UserAgentConfig

TEST_MOUSE_EMBRYIONIC_STEM_CELLS = UserAgentConfig(
    personality_description="Direct developmental biologist, will answer questions exactly and directly, is proud and certain of the research they've conducted",
    paper_description="""
INTRODUCTION: Vertebrate development comprises multiple complex morphogenetic processes that shape the embryonic body plan
through self-organization of pluripotent stem
cells and their descendants. Because mammalian embryogenesis proceeds in utero, it is difficult to study the dynamics of these processes,
including much-needed analysis at the cellular
and molecular level. Various three-dimensional
stem cell systems (“embryoids”) have been developed to circumvent this impediment. The
most advanced models of post-implantation
development achieved so far are gastruloids,
mouse embryonic stem cell (mESC)–derived
aggregates with organized gene expression
domains but lacking proper morphogenesis.
RATIONALE: To advance the current models,
we explored the usage of Matrigel, an extracellular matrix (ECM) surrogate. During embryonic development, the ECM provides
essential chemical and mechanical cues. In
vitro, lower percentages of Matrigel can drive
complex tissue morphogenesis in organoids,
which led us to use Matrigel embedding in
various media conditions to achieve higherorder embryo-like architecture in mESC-derived
aggregates.
RESULTS: We found that embedding of 96-hour
gastruloids in 5% Matrigel is sufficient to induce the formation of highly organized “trunklike structures” (TLSs), comprising the neural
tube and bilateral somites with embryo-like
polarity. This high level of self-organization was
accompanied by accumulation of the matrix
protein fibronectin at the Matrigel-TLS interface and the transcriptional up-regulation of
fibronectin-binding integrins and other cell
adhesion molecules. Chemical modulation of
signaling pathways active in the developing
mouse embryo [WNT and bone morphogenetic protein (BMP)] resulted in an excess
of somites arranged like a “bunch of grapes.”
Comparative time-resolved single-cell RNA
sequencing of TLSs and embryos revealed that
TLSs follow the same stepwise gene regulatory
programs as the mouse embryo, comprising
expression of critical developmental regulators at the right place and time. In particular,
trunk precursors known as neuromesodermal
progenitors displayed the highest differentiation potential and continuously contributed to
neural and mesodermal tissue during TLS formation. In addition, live imaging demonstrated
that the segmentation clock, required for rhythmic deposition of somites in vivo, ticks at an
embryo-like pace in TLSs. Finally, a proof-ofprinciple experiment showed that Tbx6-knockout
TLSs generate ectopic neural tubes at the expense
of somite formation, mirroring the embryonic
phenotype.
CONCLUSION: We showed that embedding of
embryonic stem cell–derived aggregates in
an ECM surrogate generates more advanced
in vitro models that are formed in a process
highly analogous to embryonic development.
Trunk-like structures represent a powerful tool
that is easily amenable to genetic, mechanical,
chemical, or other modulations. As such, we
expect them to facilitate in-depth analysis of
molecular mechanisms and signaling networks
that orchestrate embryonic development as well
as studies of the ontogeny of mutant phenotypes
in the culture dish. The scalable, tractable, and
highly accessible nature of the TLS makes it
a complementary in vitro platform for deciphering the dynamics of the molecular, cellular, and
morphogenetic processes that shape the postimplantation embryo, at an unprecedented
spatiotemporal resolution
    """,
)
