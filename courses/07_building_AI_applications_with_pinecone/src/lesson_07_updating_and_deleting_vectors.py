"""
Updating vector values

    - Keep data current
    - Optimize query performance
    - Maintain data integrity

    We call the .update() method on the index, specifying the ID of the record we wish to update and a list of new vector values. similar to when we ingest vectors, we need to ensure that this list is the same length as the index's dimensionality.
"""
index.update(
    id='1',
    values=[0.370695321, ...]
)

"""
Updating vector metadata
    To update the metadata. We'll still use the .update() method, but this time, specify the set_metadata argument. Specifically, we changed the genre from action to comedy, kept the year information as it is, as we didn't mention it in the update, and added new metadata called 'rating' with a value 5.
"""
index.update(
    id="1",
    set_metadata={"genre": "comedy", "rating": 5}
)

"""
Updating values and metadata
   We can update both the values and the metadata of a vector at the same time by combining the two arguments 
"""
index.update(
    id="1",
    values=[-0.31956, ...],
    set_metadata={"genre": "thriller", "ratings": 4}
)

"""
Deleting vectors by ID
"""
index.delete(
    ids=["1", "2"]
)

"""
Deleting vectors by-metadata
"""
index.delete(
    filter={
        "genre": {"$eq": "action"},
    }
)

"""
Deleting vectors within a namespace
    Note: Also removes the namespace!
"""
index.delete(delete_all=True, namespace='namespace1')