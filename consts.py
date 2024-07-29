from src.api.dao.graph import SubProcess, Subject, Machines, Process, Topic, DocumentChunks, Documents, Prompt, Generic

node_mapping = {
    'topic': Topic,
    'subject': Subject,
    'process': Process,
    'subprocess': SubProcess,
    'prompt': Prompt,
    'documents': Documents,
    'machines': Machines,
    'document_chunks': DocumentChunks,
    'generic': Generic
}