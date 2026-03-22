class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Target columns for multi-label classification
    # Type 1 is always single-class — ignored
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'

    # Actual column names in the CSV
    TYPE_2_COL = 'Type 2'
    TYPE_3_COL = 'Type 3'
    TYPE_4_COL = 'Type 4'

    # Minimum instances required to keep a class
    MIN_CLASS_COUNT = 5

    # Chained label separator (Design Choice 1)
    CHAIN_SEPARATOR = ' + '