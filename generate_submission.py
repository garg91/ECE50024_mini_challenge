import csv

# Path to your CSV file
csv_file_path = 'predictions_with_categories.csv'

# Initialize an empty dictionary
image_category_dict = {}

# Open and read the CSV file
with open(csv_file_path, mode='r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row

    for row in csvreader:
        image_name, category = row
        # Convert image_name to integer by removing '.jpg' and converting to int
        image_id = int(image_name.split('.')[0])
        # Add to the dictionary
        image_category_dict[image_id] = category

# print(image_category_dict)

# sort the images based on name in ascending order
sorted_image_category_dict = {k: image_category_dict[k] for k in sorted(image_category_dict)}

# print(sorted_image_category_dict)

csv_file_path = 'submission.csv'

# Open the file in write mode ('w')
with open(csv_file_path, mode='w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(['Id', 'Category'])
    # Iterate over the dictionary items and write each as a row in the file
    for key, value in sorted_image_category_dict.items():
        writer.writerow([key, value])