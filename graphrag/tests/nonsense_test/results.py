import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Read the data
df = pd.read_csv('nonsense_query_results.csv')

# Display basic info about the dataset
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# Calculate means for RAG and GraphRAG
rag_mean = df['nonsense_count_rag'].mean()
graphrag_mean = df['nonsense_count_graphrag'].mean()

print(f"\n=== OVERALL STATISTICS ===")
print(f"RAG Mean: {rag_mean:.4f}")
print(f"GraphRAG Mean: {graphrag_mean:.4f}")
print(f"Difference (RAG - GraphRAG): {rag_mean - graphrag_mean:.4f}")
print(f"GraphRAG mean is {'LESS' if graphrag_mean < rag_mean else 'GREATER'} than RAG mean")

# Statistical tests
print(f"\n=== STATISTICAL TESTS FOR MEAN COMPARISON ===")

# Check normality assumptions first
from scipy.stats import shapiro
print("Normality Tests:")
shapiro_rag = shapiro(df['nonsense_count_rag'])
shapiro_graphrag = shapiro(df['nonsense_count_graphrag'])
print(f"  RAG Shapiro-Wilk: W={shapiro_rag.statistic:.4f}, p={shapiro_rag.pvalue:.4f}")
print(f"  GraphRAG Shapiro-Wilk: W={shapiro_graphrag.statistic:.4f}, p={shapiro_graphrag.pvalue:.4f}")

# Calculate differences for paired analysis
differences = df['nonsense_count_rag'] - df['nonsense_count_graphrag']
shapiro_diff = shapiro(differences)
print(f"  Differences Shapiro-Wilk: W={shapiro_diff.statistic:.4f}, p={shapiro_diff.pvalue:.4f}")

normal_assumption = shapiro_diff.pvalue > 0.05
print(f"  Normality assumption for differences: {'Met' if normal_assumption else 'Violated'}")

print(f"\n1. PAIRED T-TEST (Parametric)")
print(f"   Null Hypothesis (H0): Mean difference = 0 (no difference between RAG and GraphRAG)")
print(f"   Alternative Hypothesis (H1): Mean difference ≠ 0 (there is a difference)")
print(f"   Assumes: Normality of differences, paired observations")

# Paired t-test (since we have paired observations for each query)
t_stat, t_pvalue = ttest_rel(df['nonsense_count_rag'], df['nonsense_count_graphrag'])
print(f"   Results:")
print(f"     t-statistic: {t_stat:.4f}")
print(f"     p-value: {t_pvalue:.4f}")
print(f"     Degrees of freedom: {len(df) - 1}")
print(f"     Mean difference: {differences.mean():.4f}")
print(f"     Standard error: {differences.std() / np.sqrt(len(df)):.4f}")
print(f"   Conclusion: {'Reject H0' if t_pvalue < 0.05 else 'Fail to reject H0'} (α = 0.05)")
print(f"   Interpretation: {'Significant difference' if t_pvalue < 0.05 else 'No significant difference'} between means")

print(f"\n2. ONE-TAILED TEST: Is GraphRAG mean < RAG mean?")
print(f"   Null Hypothesis (H0): μ_GraphRAG ≥ μ_RAG")
print(f"   Alternative Hypothesis (H1): μ_GraphRAG < μ_RAG")

# One-tailed t-test
t_stat_one, t_pvalue_one = ttest_rel(df['nonsense_count_rag'], df['nonsense_count_graphrag'], alternative='greater')
t_pvalue_one_tailed = t_pvalue_one

print(f"   One-tailed p-value: {t_pvalue_one_tailed:.4f}")
print(f"   Conclusion: {'GraphRAG mean is significantly lower' if t_pvalue_one_tailed < 0.05 and graphrag_mean < rag_mean else 'No significant evidence that GraphRAG mean is lower'}")

print(f"\n3. CONFIDENCE INTERVAL FOR MEAN DIFFERENCE")
from scipy.stats import t
alpha = 0.05
df_ci = len(differences) - 1
t_critical = t.ppf(1 - alpha/2, df_ci)
margin_error = t_critical * (differences.std() / np.sqrt(len(differences)))
ci_lower = differences.mean() - margin_error
ci_upper = differences.mean() + margin_error
print(f"   95% Confidence Interval for mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"   Interpretation: We are 95% confident the true mean difference lies in this range")
if ci_lower > 0:
    print(f"   Since CI > 0, RAG consistently has higher nonsense count than GraphRAG")
elif ci_upper < 0:
    print(f"   Since CI < 0, GraphRAG consistently has higher nonsense count than RAG")
else:
    print(f"   Since CI includes 0, there may be no meaningful difference")

# Calculate statistics by topic
topic_stats = df.groupby('topic').agg({
    'nonsense_count_rag': ['sum', 'mean', 'count'],
    'nonsense_count_graphrag': ['sum', 'mean', 'count']
}).round(4)

topic_stats.columns = ['RAG_Sum', 'RAG_Mean', 'RAG_Count', 'GraphRAG_Sum', 'GraphRAG_Mean', 'GraphRAG_Count']
topic_stats['Difference'] = topic_stats['RAG_Mean'] - topic_stats['GraphRAG_Mean']

print(f"\n=== STATISTICS BY TOPIC ===")
print(topic_stats)

# Create visualizations
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Overall comparison bar chart
ax1 = axes[0, 0]
methods = ['RAG', 'GraphRAG']
means = [rag_mean, graphrag_mean]
colors = ['#ff7f7f', '#7fbfff']

bars1 = ax1.bar(methods, means, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Overall Mean Comparison: RAG vs GraphRAG', fontsize=14, fontweight='bold')
ax1.set_ylabel('Mean Nonsense Count')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, mean_val in zip(bars1, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Sum by topic comparison
ax2 = axes[0, 1]
topics = topic_stats.index
x_pos = np.arange(len(topics))
width = 0.35

bars2_rag = ax2.bar(x_pos - width/2, topic_stats['RAG_Sum'], width,
                    label='RAG', color='#ff7f7f', alpha=0.7, edgecolor='black')
bars2_graph = ax2.bar(x_pos + width/2, topic_stats['GraphRAG_Sum'], width,
                      label='GraphRAG', color='#7fbfff', alpha=0.7, edgecolor='black')

ax2.set_title('Total Nonsense Count by Topic', fontsize=14, fontweight='bold')
ax2.set_xlabel('Topic')
ax2.set_ylabel('Total Nonsense Count')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(topics, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars2_rag, bars2_graph]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 3. Mean by topic comparison
ax3 = axes[1, 0]
bars3_rag = ax3.bar(x_pos - width/2, topic_stats['RAG_Mean'], width,
                    label='RAG', color='#ff7f7f', alpha=0.7, edgecolor='black')
bars3_graph = ax3.bar(x_pos + width/2, topic_stats['GraphRAG_Mean'], width,
                      label='GraphRAG', color='#7fbfff', alpha=0.7, edgecolor='black')

ax3.set_title('Mean Nonsense Count by Topic', fontsize=14, fontweight='bold')
ax3.set_xlabel('Topic')
ax3.set_ylabel('Mean Nonsense Count')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(topics, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars3_rag, bars3_graph]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# 4. Difference plot (RAG - GraphRAG)
ax4 = axes[1, 1]
colors_diff = ['red' if x > 0 else 'green' for x in topic_stats['Difference']]
bars4 = ax4.bar(topics, topic_stats['Difference'], color=colors_diff, alpha=0.7, edgecolor='black')

ax4.set_title('Difference in Means (RAG - GraphRAG) by Topic', fontsize=14, fontweight='bold')
ax4.set_xlabel('Topic')
ax4.set_ylabel('Difference in Mean Nonsense Count')
ax4.set_xticklabels(topics, rotation=45, ha='right')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar, diff_val in zip(bars4, topic_stats['Difference']):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (0.05 if diff_val >= 0 else -0.1),
             f'{diff_val:.3f}', ha='center',
             va='bottom' if diff_val >= 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.show()

# Additional analysis: Distribution comparison
plt.figure(figsize=(12, 5))

# Box plots
plt.subplot(1, 2, 1)
data_for_box = [df['nonsense_count_rag'], df['nonsense_count_graphrag']]
box_plot = plt.boxplot(data_for_box, labels=['RAG', 'GraphRAG'], patch_artist=True)
box_plot['boxes'][0].set_facecolor('#ff7f7f')
box_plot['boxes'][1].set_facecolor('#7fbfff')
plt.title('Distribution Comparison: RAG vs GraphRAG', fontweight='bold')
plt.ylabel('Nonsense Count')
plt.grid(axis='y', alpha=0.3)

# Histogram comparison
plt.subplot(1, 2, 2)
plt.hist(df['nonsense_count_rag'], bins=8, alpha=0.7, label='RAG', color='#ff7f7f', edgecolor='black')
plt.hist(df['nonsense_count_graphrag'], bins=8, alpha=0.7, label='GraphRAG', color='#7fbfff', edgecolor='black')
plt.title('Histogram Comparison', fontweight='bold')
plt.xlabel('Nonsense Count')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Summary conclusions
print(f"\n=== COMPREHENSIVE STATISTICAL SUMMARY ===")
print(f"DESCRIPTIVE STATISTICS:")
print(f"  RAG mean: {rag_mean:.4f} (SD: {df['nonsense_count_rag'].std():.4f})")
print(f"  GraphRAG mean: {graphrag_mean:.4f} (SD: {df['nonsense_count_graphrag'].std():.4f})")
print(f"  Mean difference: {rag_mean - graphrag_mean:.4f}")
print(f"  Median difference: {differences.median():.4f}")

print(f"\nSTATISTICAL TEST RESULTS:")
print(f"  Paired t-test p-value: {t_pvalue:.6f}")
print(f"  One-tailed test p-value: {t_pvalue_one_tailed:.6f}")

print(f"\nCONCLUSIONS:")
print(f"1. GraphRAG mean ({graphrag_mean:.4f}) is {'LOWER' if graphrag_mean < rag_mean else 'HIGHER'} than RAG mean ({rag_mean:.4f})")

improvement_percentage = ((rag_mean - graphrag_mean) / rag_mean) * 100
print(f"2. GraphRAG shows a {improvement_percentage:.1f}% {'improvement' if improvement_percentage > 0 else 'degradation'} over RAG")

print(f"3. Statistical significance at α=0.05:")
if t_pvalue < 0.001:
    significance_level = "highly significant (p < 0.001)"
elif t_pvalue < 0.01:
    significance_level = "very significant (p < 0.01)"
elif t_pvalue < 0.05:
    significance_level = "significant (p < 0.05)"
else:
    significance_level = "not significant (p ≥ 0.05)"
print(f"   Two-tailed test: {significance_level}")

if graphrag_mean < rag_mean and t_pvalue_one_tailed < 0.05:
    print(f"   One-tailed test: GraphRAG is significantly better (p = {t_pvalue_one_tailed:.6f})")
else:
    print(f"   One-tailed test: No significant evidence GraphRAG is better")

print(f"\nTOPIC-SPECIFIC ANALYSIS:")
better_topics = topic_stats[topic_stats['Difference'] > 0].index.tolist()
if better_topics:
    print(f"Topics where GraphRAG performs better (lower nonsense count):")
    for topic in better_topics:
        diff = topic_stats.loc[topic, 'Difference']
        rag_topic_mean = topic_stats.loc[topic, 'RAG_Mean']
        graphrag_topic_mean = topic_stats.loc[topic, 'GraphRAG_Mean']
        topic_improvement = (diff / rag_topic_mean) * 100
        print(f"  - {topic}:")
        print(f"    RAG: {rag_topic_mean:.3f}, GraphRAG: {graphrag_topic_mean:.3f}")
        print(f"    Improvement: {diff:.3f} ({topic_improvement:.1f}%)")
else:
    print("No topics where GraphRAG performs better than RAG")

worse_topics = topic_stats[topic_stats['Difference'] < 0].index.tolist()
if worse_topics:
    print(f"\nTopics where RAG performs better:")
    for topic in worse_topics:
        diff = abs(topic_stats.loc[topic, 'Difference'])
        print(f"  - {topic}: {diff:.3f} fewer nonsense counts with RAG")