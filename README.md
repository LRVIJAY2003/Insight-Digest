<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Knowledge Explorer</title>
    <link rel="stylesheet" href="{{ url_for('static', path='/css/styles.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', path='/css/theme-dark.css') }}" id="theme-style" disabled>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body class="{% if theme == 'dark' %}dark-mode{% endif %}">
    <div class="app-container">
        <header class="app-header">
            <div class="logo-container">
                <div class="logo">
                    <svg width="40" height="40" viewBox="0 0 100 100" fill="currentColor">
                        <rect width="100" height="100" rx="20" fill="currentColor" />
                        <rect x="50" width="50" height="100" rx="0" fill="var(--color-accent)" />
                        <circle cx="30" cy="30" r="15" fill="var(--color-background)" />
                        <circle cx="70" cy="70" r="15" fill="var(--color-background)" />
                        <path d="M30,50 L70,50" stroke="var(--color-background)" stroke-width="5" />
                        <path d="M50,30 L50,70" stroke="var(--color-background)" stroke-width="5" />
                    </svg>
                </div>
                <h1>Enterprise Knowledge Explorer</h1>
            </div>
            <div class="header-controls">
                <button id="theme-toggle" class="theme-toggle" aria-label="Toggle theme">
                    <i class="fas fa-moon"></i>
                    <i class="fas fa-sun"></i>
                </button>
            </div>
        </header>

        <main class="app-main">
            <section class="search-section">
                <div class="search-container">
                    <form action="/query" method="post" id="queryForm">
                        <div class="search-input-container">
                            <input type="text" id="query" name="query" placeholder="Ask anything about your organization..." value="{{ query if query else '' }}" required>
                            <input type="hidden" name="theme" id="theme-input" value="{{ theme if theme else 'light' }}">
                            <button type="submit" class="search-button">
                                <span id="search-button-text">Search</span>
                                <div id="search-spinner" class="spinner hidden">
                                    <i class="fas fa-circle-notch fa-spin"></i>
                                </div>
                            </button>
                        </div>
                        <div class="search-options">
                            <label class="source-option">
                                <input type="checkbox" id="source_confluence" name="source_confluence" {% if source_confluence %}checked{% endif %}>
                                <span class="checkmark"></span>
                                <span class="source-label">Confluence</span>
                            </label>
                            <label class="source-option">
                                <input type="checkbox" id="source_remedy" name="source_remedy" {% if source_remedy %}checked{% endif %}>
                                <span class="checkmark"></span>
                                <span class="source-label">Remedy</span>
                            </label>
                        </div>
                    </form>
                </div>
            </section>

            {% if error %}
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>{{ error }}</p>
            </div>
            {% endif %}

            {% if results %}
            <section class="results-section">
                {% if keywords %}
                <div class="keywords-container">
                    <div class="keyword-title">Keywords:</div>
                    <div class="keywords-list">
                        {% for keyword in keywords %}
                        <span class="keyword">{{ keyword }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}

                <div class="results-container">
                    <div class="result-box confluence-result">
                        <div class="result-header">
                            <i class="fab fa-confluence"></i>
                            <h2>Confluence</h2>
                        </div>
                        <div class="result-content">
                            {% if confluence_summary and confluence_summary != "<p>No information available.</p>" %}
                                {{ confluence_summary | safe }}
                            {% else %}
                                <p class="no-results">No relevant information found in Confluence.</p>
                            {% endif %}
                        </div>
                    </div>
                    <div class="result-box remedy-result">
                        <div class="result-header">
                            <i class="fas fa-ticket-alt"></i>
                            <h2>Remedy</h2>
                        </div>
                        <div class="result-content">
                            {% if remedy_summary and remedy_summary != "<p>No information available.</p>" %}
                                {{ remedy_summary | safe }}
                            {% else %}
                                <p class="no-results">No relevant information found in Remedy.</p>
                            {% endif %}
                        </div>
                    </div>
                </div>
                
                <div class="footer-info">
                    <div class="processing-time">
                        <i class="fas fa-clock"></i>
                        <span>Processed in {{ processing_time }} seconds</span>
                    </div>
                </div>
            </section>
            {% else %}
            <section class="welcome-section">
                <div class="welcome-container">
                    <h2>Welcome to Enterprise Knowledge Explorer</h2>
                    <p>Get instant answers from your organization's knowledge base:</p>
                    <div class="example-queries">
                        <div class="example-query-column">
                            <div class="example-query" data-query="How do I reset my password?">
                                <i class="fas fa-key"></i>
                                <span>How do I reset my password?</span>
                            </div>
                            <div class="example-query" data-query="What is the Expense submission process?">
                                <i class="fas fa-receipt"></i>
                                <span>What is the Expense submission process?</span>
                            </div>
                        </div>
                        <div class="example-query-column">
                            <div class="example-query" data-query="Compare remote work vs office policies">
                                <i class="fas fa-building"></i>
                                <span>Compare remote work vs office policies</span>
                            </div>
                            <div class="example-query" data-query="Steps to request new hardware">
                                <i class="fas fa-laptop"></i>
                                <span>Steps to request new hardware</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            {% endif %}
        </main>

        <footer class="app-footer">
            <p>&copy; {% now 'utc', '%Y' %} Enterprise Knowledge Explorer</p>
        </footer>
    </div>

    <script src="{{ url_for('static', path='/js/main.js') }}"></script>
</body>
</html>