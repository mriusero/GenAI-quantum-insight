import sqlite3


class ArxivDataLoader:
    """Handles database operations for arXiv entries."""

    CREATE_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS arxiv_entries (
        id TEXT PRIMARY KEY,
        title TEXT,
        summary TEXT,
        author TEXT,
        published DATE,
        updated DATE,
        pdf_link TEXT
    )
    """

    INSERT_QUERY = """
    INSERT INTO arxiv_entries (id, title, summary, author, published, updated, pdf_link)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    UPDATE_QUERY = """
    UPDATE arxiv_entries 
    SET title = ?, summary = ?, author = ?, published = ?, updated = ?, pdf_link = ?
    WHERE id = ?
    """

    SELECT_ENTRY_QUERY = "SELECT 1 FROM arxiv_entries WHERE id = ?"
    SELECT_UPDATED_QUERY = "SELECT updated FROM arxiv_entries WHERE id = ?"

    def __init__(self, db_name="./database/arxiv_data.db"):
        """Initialize the data loader with the specified database name."""
        self.db_name = db_name
        self.conn = None

    def __enter__(self):
        """Establish a connection to the database and create the table if it doesn't exist."""
        self.conn = sqlite3.connect(self.db_name)
        self.create_table()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def create_table(self):
        """Create the arxiv_entries table in the database."""
        with self.conn:
            self.conn.execute(self.CREATE_TABLE_QUERY)

    def entry_exists(self, arxiv_id):
        """Check if an entry with the specified ID exists in the database."""
        cursor = self.conn.execute(self.SELECT_ENTRY_QUERY, (arxiv_id,))
        return cursor.fetchone() is not None

    def update_data(self, entry):
        """Update an existing entry in the database."""
        with self.conn:
            self.conn.execute(self.UPDATE_QUERY, (
                entry['title'], entry['summary'], entry['author'],
                entry['published'], entry['updated'], entry['pdf_link'], entry['id']
            ))

    def insert_data(self, entry):
        """Insert a new entry into the database."""
        with self.conn:
            self.conn.execute(self.INSERT_QUERY, (
                entry['id'], entry['title'], entry['summary'], entry['author'],
                entry['published'], entry['updated'], entry['pdf_link']
            ))

    def parse_and_insert(self, entries):
        """Parse entries and insert or update them in the database."""
        new_entries, updated_entries = 0, 0
        for entry in entries:
            if not self.entry_exists(entry['id']):
                self.insert_data(entry)
                new_entries += 1
            else:
                existing_updated = self.conn.execute(self.SELECT_UPDATED_QUERY, (entry['id'],)).fetchone()
                if existing_updated and existing_updated[0] != entry['updated']:
                    self.update_data(entry)
                    updated_entries += 1
        return new_entries, updated_entries
