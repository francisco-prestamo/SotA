import json
import csv


def calculate_accuracy_from_json(json_file_path, output_csv_path):
    """
    Convert JSON test results to CSV with accuracy calculations.

    Args:
        json_file_path (str): Path to the input JSON file
        output_csv_path (str): Path where the CSV will be saved
    """

    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare results list
    results = []

    # Process each test case
    for test_case, test_data in data.items():
        # Calculate RAG metrics
        rag_relevants = len(test_data['rag']['relevants_returned'])
        rag_medium_relevants = len(test_data['rag']['medium_relevants_returned'])
        rag_non_relevants = len(test_data['rag']['non_relevants_returned'])
        rag_total_returned = rag_relevants + rag_medium_relevants + rag_non_relevants
        rag_accuracy = rag_relevants / rag_total_returned if rag_total_returned > 0 else 0

        # Calculate GraphRAG metrics
        graphrag_relevants = len(test_data['graphrag']['relevants_returned'])
        graphrag_medium_relevants = len(test_data['graphrag']['medium_relevants_returned'])
        graphrag_non_relevants = len(test_data['graphrag']['non_relevants_returned'])
        graphrag_total_returned = graphrag_relevants + graphrag_medium_relevants + graphrag_non_relevants
        graphrag_accuracy = graphrag_relevants / graphrag_total_returned if graphrag_total_returned > 0 else 0

        # Add to results
        results.append({
            'test_case': test_case,
            'rag_relevants': rag_relevants,
            'rag_medium_relevants': rag_medium_relevants,
            'rag_non_relevants': rag_non_relevants,
            'rag_total_returned': rag_total_returned,
            'rag_accuracy': round(rag_accuracy, 4),
            'graphrag_relevants': graphrag_relevants,
            'graphrag_medium_relevants': graphrag_medium_relevants,
            'graphrag_non_relevants': graphrag_non_relevants,
            'graphrag_total_returned': graphrag_total_returned,
            'graphrag_accuracy': round(graphrag_accuracy, 4)
        })

    # Write to CSV
    fieldnames = [
        'test_case',
        'rag_relevants', 'rag_medium_relevants', 'rag_non_relevants', 'rag_total_returned', 'rag_accuracy',
        'graphrag_relevants', 'graphrag_medium_relevants', 'graphrag_non_relevants', 'graphrag_total_returned',
        'graphrag_accuracy'
    ]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV file created successfully: {output_csv_path}")
    print(f"Processed {len(results)} test cases")

    # Print summary statistics
    avg_rag_accuracy = sum(r['rag_accuracy'] for r in results) / len(results)
    avg_graphrag_accuracy = sum(r['graphrag_accuracy'] for r in results) / len(results)

    print(f"\nSummary:")
    print(f"Average RAG accuracy: {avg_rag_accuracy:.4f}")
    print(f"Average GraphRAG accuracy: {avg_graphrag_accuracy:.4f}")

    return results


def calculate_accuracy_from_json_string(json_string, output_csv_path):
    """
    Alternative function that takes JSON as a string instead of file path.
    Useful if you have the JSON data as a string variable.
    """
    data = json.loads(json_string)

    # Same processing logic as above
    results = []

    for test_case, test_data in data.items():
        rag_relevants = len(test_data['rag']['relevants_returned'])
        rag_medium_relevants = len(test_data['rag']['medium_relevants_returned'])
        rag_non_relevants = len(test_data['rag']['non_relevants_returned'])
        rag_total_returned = rag_relevants + rag_medium_relevants + rag_non_relevants
        rag_accuracy = rag_relevants / rag_total_returned if rag_total_returned > 0 else 0

        graphrag_relevants = len(test_data['graphrag']['relevants_returned'])
        graphrag_medium_relevants = len(test_data['graphrag']['medium_relevants_returned'])
        graphrag_non_relevants = len(test_data['graphrag']['non_relevants_returned'])
        graphrag_total_returned = graphrag_relevants + graphrag_medium_relevants + graphrag_non_relevants
        graphrag_accuracy = graphrag_relevants / graphrag_total_returned if graphrag_total_returned > 0 else 0

        results.append({
            'test_case': test_case,
            'rag_relevants': rag_relevants,
            'rag_medium_relevants': rag_medium_relevants,
            'rag_non_relevants': rag_non_relevants,
            'rag_total_returned': rag_total_returned,
            'rag_accuracy': round(rag_accuracy, 4),
            'graphrag_relevants': graphrag_relevants,
            'graphrag_medium_relevants': graphrag_medium_relevants,
            'graphrag_non_relevants': graphrag_non_relevants,
            'graphrag_total_returned': graphrag_total_returned,
            'graphrag_accuracy': round(graphrag_accuracy, 4)
        })

    fieldnames = [
        'test_case',
        'rag_relevants', 'rag_medium_relevants', 'rag_non_relevants', 'rag_total_returned', 'rag_accuracy',
        'graphrag_relevants', 'graphrag_medium_relevants', 'graphrag_non_relevants', 'graphrag_total_returned',
        'graphrag_accuracy'
    ]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results


# Example usage:
if __name__ == "__main__":
    # Method 1: From JSON file
    calculate_accuracy_from_json('test_results.json', 'accuracy_results.csv')

    # Method 2: From JSON string (if you have JSON as a string variable)
    # json_data = '{"PLAIN-2600": {...}, "PLAIN-2760": {...}}'
    # calculate_accuracy_from_json_string(json_data, 'accuracy_results.csv')