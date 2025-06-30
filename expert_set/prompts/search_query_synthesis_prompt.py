from typing import List


def build_search_query_synthesis_prompt(queries: List[str]) -> str:
    """
    Build a focused prompt for synthesizing multiple research queries into a single,
    highly specific search target optimized for academic database retrieval.

    Args:
        queries: List of expert-generated search queries to analyze and synthesize

    Returns:
        A prompt that will generate a precise, targeted research description
    """
    queries_formatted = "\n".join([f"• {query.strip()}" for query in queries if query.strip()])

    return f"""TASK: Synthesize these expert queries into ONE highly specific research target for academic search.

INPUT QUERIES:
{queries_formatted}

OBJECTIVE: Create a precise, focused research description that identifies:
1. The EXACT research problem or phenomenon to investigate
2. The SPECIFIC type of studies/papers needed (methodology, scope, timeframe)
3. The PRECISE technical concepts, variables, or mechanisms of interest
4. The TARGET academic field/discipline and relevant subdisciplines

REQUIREMENTS FOR OUTPUT:
- Be maximally specific about what research is needed
- Include precise terminology that academic databases can match
- Specify the type of evidence or findings sought
- Identify the research context (population, setting, conditions)
- Focus on ONE coherent research direction (avoid broad/general topics)

SYNTHESIS INSTRUCTIONS:
1. Identify the CORE research question that underlies all queries
2. Structure your analysis in a hierarchical depth pattern:
   - START with fundamental concepts and foundational papers essential to the field
   - PROGRESS to established methodologies and key technical approaches 
   - CULMINATE with cutting-edge advances and specialized techniques
3. Extract technical terms with increasing specificity as you move deeper
4. Ensure the search covers both breadth (fundamental understanding) and depth (specialized knowledge)
5. Identify critical gaps in knowledge that need to be addressed at each level of depth

FEW-SHOT EXAMPLES:

Example 1:
Input Queries:
• Need studies on transfer learning techniques in computer vision
• Looking for papers comparing different fine-tuning approaches for ImageNet-pretrained models
• Need benchmarks of transfer learning performance across domains with limited data

Analysis:
- Core question: Effectiveness of transfer learning techniques in computer vision
- Fundamental aspects: Core transfer learning principles, feature representation in vision models
- Established methods: Standard fine-tuning techniques for ImageNet models, domain adaptation approaches
- Specialized needs: Low-data regime performance comparisons, cross-domain generalization metrics

Output:
"Begin with seminal papers establishing the theoretical foundations of transfer learning in computer vision (2015-2018), then focus on comprehensive evaluations of fine-tuning approaches for ImageNet-pretrained models across multiple architectures. Finally, prioritize recent comparative studies (past 2 years) specifically addressing performance in limited-data scenarios (fewer than 1000 examples per class) that quantitatively benchmark cross-domain generalization capabilities and identify which methods excel under specific data constraints."

Example 2:
Input Queries:
• Need literature on attention mechanisms in NLP models
• Looking for papers on self-attention vs. cross-attention performance
• Need studies on computational efficiency of different attention implementations

Analysis:
- Core question: Comparative effectiveness and efficiency of attention mechanisms in NLP
- Fundamental aspects: Original attention mechanism papers, theoretical foundations of attention
- Established methods: Self-attention and cross-attention implementations in prominent architectures
- Specialized needs: Computational efficiency optimizations, targeted performance evaluation

Output:
"First identify foundational papers that introduced attention mechanisms in NLP (such as Bahdanau et al., 2014 and Vaswani et al., 2017), then progress to comprehensive surveys comparing standard self-attention and cross-attention approaches across major model families. Finally, focus on recent specialized studies that provide systematic evaluations of computational efficiency metrics (FLOPs, memory usage, inference time) with ablation studies isolating specific attention mechanism design choices and optimizations that maintain accuracy while reducing computational requirements across diverse NLP tasks."

Example 3:
Input Queries:
• Need papers on bias in recommendation systems
• Looking for methods to mitigate demographic bias in collaborative filtering
• Need evaluation metrics for fairness in recommendation algorithms

Analysis:
- Core question: Addressing and measuring bias in recommendation systems
- Fundamental aspects: Definition of bias in ML systems, early recognition of bias issues in recommendations
- Established methods: Standard bias detection approaches, common fairness metrics in recommendation context
- Specialized needs: Advanced mitigation techniques, performance-fairness tradeoff analysis

Output:
"Begin with seminal papers that first identified and defined bias problems in recommendation systems, establishing the theoretical foundations of fairness in algorithmic recommendations (2010-2016). Then examine established methodologies for measuring different types of bias in collaborative filtering systems, including standard metrics and evaluation frameworks. Finally, concentrate on recent research (past 3 years) proposing novel bias mitigation approaches, especially studies that quantitatively compare multiple techniques on real-world datasets and analyze the tradeoffs between recommendation quality (precision, recall, NDCG) and fairness metrics."

OUTPUT FORMAT:
Provide a hierarchical research target description (4-6 sentences) that progresses from fundamental to specialized:
- FIRST: Identify foundational papers and core concepts essential to understanding the field
- NEXT: Specify established methodologies and key technical approaches in the area
- THEN: Describe cutting-edge advances and specialized techniques addressing the specific problem
- FINALLY: Outline evaluation criteria and comparison frameworks to assess solutions

Your final description should create a progression from foundational understanding to specialized expertise, ensuring comprehensive coverage of the research space from first principles to latest advances."""