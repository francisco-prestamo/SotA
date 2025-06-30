import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, t
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Read the data
csv_file = 'accuracy_results.csv'
df = pd.read_csv(csv_file)

# Display basic info about the dataset
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# Calculate means for RAG and GraphRAG
rag_mean = df['rag_accuracy'].mean()
graphrag_mean = df['graphrag_accuracy'].mean()

print(f"\n=== OVERALL STATISTICS ===")
print(f"RAG Mean Accuracy: {rag_mean:.4f}")
print(f"GraphRAG Mean Accuracy: {graphrag_mean:.4f}")
print(f"Difference (GraphRAG - RAG): {graphrag_mean - rag_mean:.4f}")
print(f"GraphRAG mean is {'HIGHER' if graphrag_mean > rag_mean else 'LOWER'} than RAG mean")

# Statistical tests
print(f"\n=== STATISTICAL TESTS FOR MEAN COMPARISON ===")

# Check normality assumptions first
print("Normality Tests:")
shapiro_rag = shapiro(df['rag_accuracy'])
shapiro_graphrag = shapiro(df['graphrag_accuracy'])
print(f"  RAG Shapiro-Wilk: W={shapiro_rag.statistic:.4f}, p={shapiro_rag.pvalue:.4f}")
print(f"  GraphRAG Shapiro-Wilk: W={shapiro_graphrag.statistic:.4f}, p={shapiro_graphrag.pvalue:.4f}")

differences = df['graphrag_accuracy'] - df['rag_accuracy']
shapiro_diff = shapiro(differences)
print(f"  Differences Shapiro-Wilk: W={shapiro_diff.statistic:.4f}, p={shapiro_diff.pvalue:.4f}")
normal_assumption = shapiro_diff.pvalue > 0.05
print(f"  Normality assumption for differences: {'Met' if normal_assumption else 'Violated'}")

print(f"\n1. TWO-TAILED TEST: Are means equal?")
print(f"   Null Hypothesis (H0): Mean difference = 0 (no difference between RAG and GraphRAG)")
print(f"   Alternative Hypothesis (H1): Mean difference ≠ 0 (there is a difference)")
t_stat, t_pvalue = ttest_rel(df['graphrag_accuracy'], df['rag_accuracy'])
print(f"   Results:")
print(f"     t-statistic: {t_stat:.4f}")
print(f"     p-value: {t_pvalue:.4f}")
print(f"     Degrees of freedom: {len(df) - 1}")
print(f"     Mean difference: {differences.mean():.4f}")
print(f"     Standard error: {differences.std() / np.sqrt(len(df)):.4f}")
print(f"   Conclusion: {'Reject H0' if t_pvalue < 0.05 else 'Fail to reject H0'} (α = 0.05)")
print(f"   Interpretation: {'Significant difference' if t_pvalue < 0.05 else 'No significant difference'} between means")

print(f"\n2. ONE-TAILED TEST: Is GraphRAG accuracy > RAG accuracy?")
print(f"   Null Hypothesis (H0): μ_GraphRAG ≤ μ_RAG")
print(f"   Alternative Hypothesis (H1): μ_GraphRAG > μ_RAG")
t_stat_one, t_pvalue_one = ttest_rel(df['graphrag_accuracy'], df['rag_accuracy'], alternative='greater')
print(f"   One-tailed p-value: {t_pvalue_one:.4f}")
print(f"   Conclusion: {'GraphRAG mean is significantly higher' if t_pvalue_one < 0.05 and graphrag_mean > rag_mean else 'No significant evidence that GraphRAG mean is higher'}")

# Optional: Visualization
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Overall comparison bar chart
ax1 = axes[0]
methods = ['RAG', 'GraphRAG']
means = [rag_mean, graphrag_mean]
colors = ['#ff7f7f', '#7fbfff']
bars1 = ax1.bar(methods, means, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Overall Mean Accuracy: RAG vs GraphRAG', fontsize=14, fontweight='bold')
ax1.set_ylabel('Mean Accuracy')
ax1.grid(axis='y', alpha=0.3)
for bar, mean_val in zip(bars1, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Distribution comparison
ax2 = axes[1]
data_for_box = [df['rag_accuracy'], df['graphrag_accuracy']]
box_plot = ax2.boxplot(data_for_box, labels=['RAG', 'GraphRAG'], patch_artist=True)
box_plot['boxes'][0].set_facecolor('#ff7f7f')
box_plot['boxes'][1].set_facecolor('#7fbfff')
ax2.set_title('Distribution Comparison: RAG vs GraphRAG', fontweight='bold')
ax2.set_ylabel('Accuracy')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Summary conclusions
print(f"\n=== SUMMARY ===")
print(f"RAG mean: {rag_mean:.4f} (SD: {df['rag_accuracy'].std():.4f})")
print(f"GraphRAG mean: {graphrag_mean:.4f} (SD: {df['graphrag_accuracy'].std():.4f})")
print(f"Mean difference: {graphrag_mean - rag_mean:.4f}")
print(f"Paired t-test p-value: {t_pvalue:.6f}")
print(f"One-tailed test p-value: {t_pvalue_one:.6f}")
if graphrag_mean > rag_mean and t_pvalue_one < 0.05:
    print(f"GraphRAG is significantly better (p = {t_pvalue_one:.6f})")
else:
    print(f"No significant evidence GraphRAG is better")
