/**
 * Main JavaScript for Enterprise Knowledge Explorer
 */

document.addEventListener('DOMContentLoaded', function() {
    // Theme toggle functionality
    setupThemeToggle();
    
    // Example query functionality
    setupExampleQueries();
    
    // Form submission with loading indicator
    setupFormSubmission();
    
    // Source checkboxes validation
    setupSourceValidation();
    
    // Initialize scrollable areas
    setupScrollableAreas();
});

/**
 * Set up theme toggle between light and dark mode
 */
function setupThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    const themeStyle = document.getElementById('theme-style');
    const themeInput = document.getElementById('theme-input');
    const body = document.body;
    
    // Check for stored theme preference
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme === 'dark') {
        enableDarkMode();
    }
    
    // Toggle theme on button click
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            if (body.classList.contains('dark-mode')) {
                disableDarkMode();
            } else {
                enableDarkMode();
            }
        });
    }
    
    function enableDarkMode() {
        body.classList.add('dark-mode');
        themeStyle.removeAttribute('disabled');
        localStorage.setItem('theme', 'dark');
        if (themeInput) themeInput.value = 'dark';
    }
    
    function disableDarkMode() {
        body.classList.remove('dark-mode');
        themeStyle.setAttribute('disabled', true);
        localStorage.setItem('theme', 'light');
        if (themeInput) themeInput.value = 'light';
    }
}

/**
 * Set up example query functionality
 */
function setupExampleQueries() {
    const exampleQueries = document.querySelectorAll('.example-query');
    const queryInput = document.getElementById('query');
    const queryForm = document.getElementById('queryForm');
    
    exampleQueries.forEach(query => {
        query.addEventListener('click', function() {
            const queryText = this.getAttribute('data-query');
            if (queryInput && queryText) {
                queryInput.value = queryText;
                queryInput.focus();
                
                // Optionally submit the form automatically
                // if (queryForm) queryForm.submit();
            }
        });
    });
}

/**
 * Set up form submission with loading indicator
 */
function setupFormSubmission() {
    const queryForm = document.getElementById('queryForm');
    const searchButton = document.querySelector('.search-button');
    const searchButtonText = document.getElementById('search-button-text');
    const searchSpinner = document.getElementById('search-spinner');
    
    if (queryForm) {
        queryForm.addEventListener('submit', function() {
            // Show loading state
            if (searchButton) searchButton.setAttribute('disabled', 'disabled');
            if (searchButtonText) searchButtonText.textContent = 'Searching...';
            if (searchSpinner) searchSpinner.classList.remove('hidden');
        });
    }
}

/**
 * Ensure at least one source is selected
 */
function setupSourceValidation() {
    const confluenceCheckbox = document.getElementById('source_confluence');
    const remedyCheckbox = document.getElementById('source_remedy');
    const queryForm = document.getElementById('queryForm');
    
    function validateSourceSelection() {
        if (!confluenceCheckbox.checked && !remedyCheckbox.checked) {
            confluenceCheckbox.setCustomValidity('Please select at least one source');
        } else {
            confluenceCheckbox.setCustomValidity('');
        }
    }
    
    if (confluenceCheckbox && remedyCheckbox) {
        confluenceCheckbox.addEventListener('change', validateSourceSelection);
        remedyCheckbox.addEventListener('change', validateSourceSelection);
        
        // Initial validation
        validateSourceSelection();
    }
    
    // Before form submission, make sure at least one is checked
    if (queryForm) {
        queryForm.addEventListener('submit', function(event) {
            if (!confluenceCheckbox.checked && !remedyCheckbox.checked) {
                event.preventDefault();
                alert('Please select at least one source (Confluence or Remedy)');
            }
        });
    }
}

/**
 * Setup scrollable areas for better mobile experience
 */
function setupScrollableAreas() {
    const resultContents = document.querySelectorAll('.result-content');
    
    // Add touch scrolling for mobile devices
    resultContents.forEach(content => {
        content.addEventListener('touchstart', function(e) {
            // Store the initial touch position
            this.startY = e.touches[0].clientY;
        }, { passive: true });
        
        content.addEventListener('touchmove', function(e) {
            if (!this.startY) return;
            
            const touchY = e.touches[0].clientY;
            const scrollTop = this.scrollTop;
            const scrollHeight = this.scrollHeight;
            const height = this.offsetHeight;
            
            // Check if at the top or bottom of the scrollable area
            const isAtTop = scrollTop === 0 && touchY > this.startY;
            const isAtBottom = scrollTop + height >= scrollHeight && touchY < this.startY;
            
            // If at the edge, prevent default to avoid body scrolling
            if (isAtTop || isAtBottom) {
                e.preventDefault();
            }
        }, { passive: false });
        
        content.addEventListener('touchend', function() {
            // Reset the touch position
            this.startY = null;
        }, { passive: true });
    });
}

/**
 * API client functions for programmatic access
 */
class ApiClient {
    /**
     * Send a query to the RAG system API
     * @param {string} query - The query text
     * @param {Array} sources - Sources to search (optional)
     * @returns {Promise} - Promise resolving to the API response
     */
    static async query(query, sources = null) {
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    sources: sources
                }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error querying the API:', error);
            throw error;
        }
    }
    
    /**
     * Clear the system cache
     * @returns {Promise} - Promise resolving to the API response
     */
    static async clearCache() {
        try {
            const response = await fetch('/clear-cache', {
                method: 'GET',
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error clearing cache:', error);
            throw error;
        }
    }
    
    /**
     * Get system information
     * @returns {Promise} - Promise resolving to the API response
     */
    static async getSystemInfo() {
        try {
            const response = await fetch('/system-info', {
                method: 'GET',
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error getting system info:', error);
            throw error;
        }
    }
}

// Make available globally for console usage
window.ApiClient = ApiClient;