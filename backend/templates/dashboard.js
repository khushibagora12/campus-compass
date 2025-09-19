const messageDiv = document.getElementById('message');
        const handleFormSubmit = async (formId, event) => {
            event.preventDefault();
            const form = document.getElementById(formId);
            const formData = new FormData(form);
            const action = (formId === 'uploadForm') ? '/admin/upload' : '/admin/ingest';
            
            messageDiv.className = '';
            messageDiv.textContent = 'Processing...';
            messageDiv.style.display = 'block';

            try {
                const response = await fetch(action, { method: 'POST', body: formData });
                const result = await response.json();
                if (response.ok) {
                    messageDiv.textContent = result.message;
                    messageDiv.className = 'success';
                } else {
                    messageDiv.textContent = result.error || 'An unknown error occurred.';
                    messageDiv.className = 'error';
                }
            } catch (error) {
                messageDiv.textContent = 'Network error. Please try again.';
                messageDiv.className = 'error';
            }
        };
        document.getElementById('uploadForm').addEventListener('submit', (e) => handleFormSubmit('uploadForm', e));
        document.getElementById('ingestForm').addEventListener('submit', (e) => handleFormSubmit('ingestForm', e));