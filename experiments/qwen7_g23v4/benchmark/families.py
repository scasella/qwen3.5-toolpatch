"""Semantic families are operations, not domain-noun paraphrases. Fixed before discovery."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Family:
    id: str
    split: str
    operations: tuple[tuple[str,str], ...]
    fields: tuple[tuple[str,str,str], ...]
# Each field: name, JSON type, semantic description. Two competing operations per family.
DATA = [
('record_retrieval','discovery', [('search_records','Find records whose text matches a query'),('fetch_record','Retrieve one record by its exact identifier')], [('key','string','Search query or exact record identifier, as requested')]),
('collection_size','discovery',[('list_items','Return the members of a collection'),('count_items','Return only the number of members of a collection')],[('collection','string','Identifier of the collection')]),
('transfer_direction','discovery',[('move_item','Move an item from a source container into a destination container'),('copy_item','Copy an item into a destination while retaining it at the source')],[('item','string','Item identifier'),('source','string','Container the item comes from'),('destination','string','Container the item goes to')]),
('sort_direction','discovery',[('sort_ascending','Sort a stored sequence from smallest to largest'),('sort_descending','Sort a stored sequence from largest to smallest')],[('sequence','string','Stored sequence identifier')]),
('reservation_lifecycle','discovery',[('create_reservation','Reserve a resource for a slot'),('cancel_reservation','Cancel the reservation for a resource and slot')],[('resource','string','Resource identifier'),('slot','integer','Slot number')]),
('unit_conversion','discovery',[('meters_to_feet','Convert a length in meters to feet'),('feet_to_meters','Convert a length in feet to meters')],[('length','integer','Numeric length in the input unit')]),
('access_control','discovery',[('grant_access','Give a member access to a group'),('revoke_access','Remove a member access to a group')],[('member','string','Member identifier'),('group','string','Group identifier')]),
('substring_position','discovery',[('first_match','Find the first position of a term in stored text'),('last_match','Find the last position of a term in stored text')],[('document','string','Stored document identifier'),('term','string','Text to locate')]),
('queue_end','discovery',[('enqueue','Put a value at the back of a queue'),('prepend_queue','Put a value at the front of a queue')],[('queue','string','Queue identifier'),('value','string','Value to insert')]),
('interval_boundary','discovery',[('closed_interval','Select values including both interval endpoints'),('open_interval','Select values excluding both interval endpoints')],[('dataset','string','Dataset identifier'),('lower','integer','Lower bound'),('upper','integer','Upper bound')]),
('record_mutation','discovery',[('replace_value','Replace an existing field value'),('append_value','Append text to an existing field value')],[('record','string','Record identifier'),('value','string','Replacement or appended text')]),
('graph_direction','discovery',[('outgoing_neighbors','List vertices reached by edges leaving a vertex'),('incoming_neighbors','List vertices with edges entering a vertex')],[('vertex','string','Vertex identifier')]),
('numeric_aggregation','discovery',[('sum_values','Return the sum of stored values'),('mean_values','Return the arithmetic mean of stored values')],[('series','string','Stored series identifier')]),
('clock_shift','discovery',[('advance_clock','Move a synthetic clock forward by minutes'),('rewind_clock','Move a synthetic clock backward by minutes')],[('clock','string','Clock identifier'),('minutes','integer','Number of minutes')]),
('text_case','discovery',[('uppercase_text','Change stored text to uppercase'),('lowercase_text','Change stored text to lowercase')],[('text','string','Stored text identifier')]),
('archive_state','discovery',[('archive_record','Mark a record archived'),('restore_record','Restore an archived record to active state')],[('record','string','Record identifier')]),
('set_membership','discovery',[('set_union','Return members in either set'),('set_intersection','Return members common to both sets')],[('left','string','First set identifier'),('right','string','Second set identifier')]),
('numeric_comparison','discovery',[('above_threshold','Select stored numbers strictly above a threshold'),('below_threshold','Select stored numbers strictly below a threshold')],[('series','string','Series identifier'),('threshold','integer','Numeric threshold')]),
('job_priority','selection',[('increase_priority','Increase a job priority by a numeric amount'),('decrease_priority','Decrease a job priority by a numeric amount')],[('job','string','Job identifier'),('amount','integer','Priority adjustment')]),
('string_padding','selection',[('pad_left','Add padding at the beginning of stored text'),('pad_right','Add padding at the end of stored text')],[('text','string','Stored text identifier'),('width','integer','Target total width')]),
('version_comparison','selection',[('version_diff','Show differences between two versions'),('version_equal','Return whether two versions are identical')],[('old','string','Earlier version identifier'),('new','string','Later version identifier')]),
('inventory_adjustment','selection',[('add_stock','Add units to inventory'),('remove_stock','Remove units from inventory')],[('sku','string','Product code, preserving leading zeros'),('units','integer','Number of units')]),
('tree_scope','selection',[('direct_children','List immediate children of a tree node'),('all_descendants','List every descendant of a tree node')],[('node','string','Tree node identifier')]),
('subscription','selection',[('subscribe','Start receiving events on a topic'),('unsubscribe','Stop receiving events on a topic')],[('subscriber','string','Subscriber identifier'),('topic','string','Topic identifier')]),
('matrix_axis','final',[('row_slice','Extract a matrix row'),('column_slice','Extract a matrix column')],[('matrix','string','Stored matrix identifier'),('index','integer','Zero based index')]),
('edge_mutation','final',[('connect_nodes','Create a directed edge'),('disconnect_nodes','Remove a directed edge')],[('source','string','Origin vertex'),('destination','string','Target vertex')]),
('document_redaction','final',[('redact_term','Hide a term in a stored document'),('highlight_term','Highlight a term in a stored document')],[('document','string','Document identifier'),('term','string','Term to process')]),
('filter_parity','final',[('even_values','Select even stored integers'),('odd_values','Select odd stored integers')],[('series','string','Series identifier')]),
('event_window','final',[('events_before','Retrieve events earlier than a timestamp'),('events_after','Retrieve events later than a timestamp')],[('stream','string','Event stream identifier'),('timestamp','integer','Synthetic timestamp')]),
('encoding_direction','final',[('encode_base64','Encode stored bytes as base64'),('decode_base64','Decode stored base64 as bytes')],[('blob','string','Stored blob identifier')]),
('ranking','final',[('top_k','Return the largest k stored values'),('bottom_k','Return the smallest k stored values')],[('series','string','Series identifier'),('k','integer','Number of values')]),
('coordinate_transform','final',[('rotate_clockwise','Rotate a point clockwise by degrees'),('rotate_counterclockwise','Rotate a point counterclockwise by degrees')],[('point','string','Stored point identifier'),('degrees','integer','Angle in degrees')]),
('key_value_inverse','final',[('lookup_value','Retrieve a value given a key'),('lookup_key','Retrieve a key given a value')],[('table','string','Table identifier'),('query','string','Key or value to look up')]),
('schedule_offset','final',[('delay_event','Schedule an event later by minutes'),('expedite_event','Schedule an event earlier by minutes')],[('event','string','Event identifier'),('minutes','integer','Adjustment in minutes')]),
('checksum','final',[('compute_checksum','Compute a checksum of stored data'),('verify_checksum','Check stored data against its saved checksum')],[('data','string','Stored data identifier')]),
('partition','final',[('split_partition','Split a partition into a specified number of parts'),('merge_partitions','Merge a specified number of adjacent partitions')],[('partition','string','Starting partition identifier'),('parts','integer','Number of parts')]),
]
FAMILIES = [Family(i,s,tuple(o),tuple(f)) for i,s,o,f in DATA]
assert len(FAMILIES)==36 and len({f.id for f in FAMILIES})==36
