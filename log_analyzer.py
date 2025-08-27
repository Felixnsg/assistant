import re
import os
import sys
import datetime
from collections import defaultdict, Counter

class SimpleLogAnalyzer:
    """Simplified analyzer for FelixTrackingClient logs focusing on errors and connection issues"""
    
    def __init__(self, log_file_path):
        """Initialize with path to log file"""
        self.log_file_path = log_file_path
        self.errors = []
        self.warnings = []
        self.connection_issues = []
        self.reconnect_events = []
        self.task_failures = []
        
        # Regular expression for parsing log lines
        self.log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([\w\.]+) - (\w+) - (.*)'
        )
        
        # Stats
        self.error_count = 0
        self.warning_count = 0
        self.connection_attempts = 0
        self.successful_connections = 0
        self.connection_failures = 0
        self.components_with_errors = Counter()
        
        # Runtime tracking
        self.first_timestamp = None
        self.last_timestamp = None
        self.sessions = []
        self.current_session = None
        
    def parse_log(self):
        """Parse the log file and extract errors and connection issues"""
        print(f"\nParsing log file: {self.log_file_path}")
        print("This may take a moment for large log files...\n")
        
        try:
            line_count = 0
            with open(self.log_file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    try:
                        # Parse the line with regex
                        match = self.log_pattern.match(line.strip())
                        if match:
                            timestamp_str, component, level, message = match.groups()
                            try:
                                timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                                
                                # Track first and last timestamps for runtime calculation
                                if self.first_timestamp is None or timestamp < self.first_timestamp:
                                    self.first_timestamp = timestamp
                                if self.last_timestamp is None or timestamp > self.last_timestamp:
                                    self.last_timestamp = timestamp
                                
                            except ValueError:
                                timestamp = datetime.datetime.now()
                            
                            # Track session boundaries
                            if "Initializing Lightweight FelixTrackingClient" in message:
                                # New session started
                                self.current_session = {
                                    'start_time': timestamp,
                                    'end_time': None,
                                    'errors': 0,
                                    'warnings': 0,
                                    'connection_established': False
                                }
                                self.sessions.append(self.current_session)
                            elif self.current_session and ("FelixTrackingClient run method finished" in message or 
                                                         "Program interrupted by user" in message):
                                # Session ended
                                self.current_session['end_time'] = timestamp
                            
                            # Track connection success in current session
                            if self.current_session and "Connection successful!" in message:
                                self.current_session['connection_established'] = True
                                
                            # Track errors
                            if level == 'ERROR':
                                self.error_count += 1
                                self.errors.append({
                                    'timestamp': timestamp,
                                    'component': component,
                                    'message': message,
                                    'line_number': line_num
                                })
                                self.components_with_errors[component] += 1
                                if self.current_session:
                                    self.current_session['errors'] += 1
                                
                            # Track warnings
                            elif level == 'WARNING':
                                self.warning_count += 1
                                self.warnings.append({
                                    'timestamp': timestamp,
                                    'component': component,
                                    'message': message,
                                    'line_number': line_num
                                })
                                if self.current_session:
                                    self.current_session['warnings'] += 1
                            
                            # Track connection-related issues
                            if any(keyword in message for keyword in 
                                  ['Connection closed', 'Connection refused', 'connection error', 
                                   'error during send', 'websocket error']):
                                self.connection_issues.append({
                                    'timestamp': timestamp,
                                    'level': level,
                                    'component': component,
                                    'message': message,
                                    'line_number': line_num
                                })
                                
                            # Track connection attempts and successes/failures
                            if "Connection attempt" in message and "to ws://" in message:
                                self.connection_attempts += 1
                            elif "Connection successful!" in message:
                                self.successful_connections += 1
                            elif ("Connection closed" in message or "Connection refused" in message or
                                  "connect error" in message.lower()):
                                self.connection_failures += 1
                                
                            # Track reconnection events
                            if "Will retry" in message or "Reconnect condition met" in message:
                                self.reconnect_events.append({
                                    'timestamp': timestamp,
                                    'component': component,
                                    'message': message,
                                    'line_number': line_num
                                })
                                
                            # Track task failures
                            if "Task '" in message and "failed" in message:
                                self.task_failures.append({
                                    'timestamp': timestamp,
                                    'component': component,
                                    'message': message,
                                    'line_number': line_num
                                })
                    except Exception as e:
                        # Skip problematic lines
                        continue
            
            print(f"Successfully parsed {line_count} log lines")
            print(f"Found {self.error_count} errors and {len(self.connection_issues)} connection issues")
            
        except FileNotFoundError:
            print(f"Error: Log file not found at {self.log_file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing log file: {str(e)}")
            sys.exit(1)
    
    def show_quick_summary(self):
        """Show a brief summary of errors and connection issues"""
        print("\n" + "="*60)
        print("QUICK SUMMARY")
        print("="*60)
        
        print(f"\nTotal Errors: {self.error_count}")
        print(f"Total Warnings: {self.warning_count}")
        
        print(f"\nConnection Statistics:")
        print(f"  - Connection Attempts: {self.connection_attempts}")
        print(f"  - Successful Connections: {self.successful_connections}")
        print(f"  - Connection Failures: {self.connection_failures}")
        print(f"  - Reconnect Events: {len(self.reconnect_events)}")
        
        print(f"\nTask Failures: {len(self.task_failures)}")
        
        # Show components with the most errors
        if self.components_with_errors:
            print("\nComponents with errors:")
            for component, count in self.components_with_errors.most_common(5):
                print(f"  - {component}: {count} errors")
                
        # Show time range if we have timestamps
        if self.first_timestamp and self.last_timestamp:
            print(f"\nLog Time Range: {self.first_timestamp} to {self.last_timestamp}")
            total_runtime = (self.last_timestamp - self.first_timestamp).total_seconds()
            print(f"Total Runtime: {self.format_duration(total_runtime)}")
    
    def show_all_errors(self):
        """Show all errors chronologically"""
        print("\n" + "="*60)
        print("ALL ERRORS")
        print("="*60)
        
        if not self.errors:
            print("\nNo errors found in the log.")
            return
            
        # Sort errors by timestamp
        sorted_errors = sorted(self.errors, key=lambda x: x['timestamp'])
        
        for i, error in enumerate(sorted_errors, 1):
            time_str = error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{i}. [Line {error['line_number']}] {time_str} - {error['component']}")
            print(f"   {error['message']}")
            
            # Add separator between errors for readability
            if i < len(sorted_errors):
                print("-" * 40)
    
    def show_error_categories(self):
        """Group similar errors and show categories"""
        print("\n" + "="*60)
        print("ERROR CATEGORIES")
        print("="*60)
        
        if not self.errors:
            print("\nNo errors found in the log.")
            return
            
        # Group similar errors
        error_categories = defaultdict(list)
        for error in self.errors:
            # Use the first 50 chars as a simple categorization key
            key = error['message'][:50]
            error_categories[key].append(error)
        
        # Display categories
        print(f"\nFound {len(error_categories)} distinct error categories:")
        
        for i, (key, errors) in enumerate(sorted(error_categories.items(), 
                                                key=lambda x: len(x[1]), 
                                                reverse=True), 1):
            first_occurrence = min(error['timestamp'] for error in errors)
            last_occurrence = max(error['timestamp'] for error in errors)
            
            first_time = first_occurrence.strftime('%Y-%m-%d %H:%M:%S')
            last_time = last_occurrence.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n{i}. Error Type: {key}...")
            print(f"   Count: {len(errors)}")
            print(f"   First occurrence: {first_time}")
            print(f"   Last occurrence: {last_time}")
            print(f"   Components affected: {', '.join(set(error['component'] for error in errors))}")
    
    def show_connection_issues(self):
        """Show all connection-related issues"""
        print("\n" + "="*60)
        print("CONNECTION ISSUES")
        print("="*60)
        
        if not self.connection_issues and not self.reconnect_events:
            print("\nNo connection issues found in the log.")
            return
            
        print("\nConnection Attempts vs Results:")
        print(f"  - Total attempts: {self.connection_attempts}")
        print(f"  - Successful: {self.successful_connections}")
        print(f"  - Failed: {self.connection_failures}")
        success_rate = (self.successful_connections / self.connection_attempts * 100) if self.connection_attempts > 0 else 0
        print(f"  - Success rate: {success_rate:.1f}%")
        
        # Show reconnection events
        if self.reconnect_events:
            print(f"\nReconnection Events ({len(self.reconnect_events)}):")
            for i, event in enumerate(sorted(self.reconnect_events, key=lambda x: x['timestamp']), 1):
                time_str = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n{i}. {time_str} - {event['component']}")
                print(f"   {event['message']}")
        
        # Show connection issues details
        if self.connection_issues:
            print(f"\nDetailed Connection Issues ({len(self.connection_issues)}):")
            for i, issue in enumerate(sorted(self.connection_issues, key=lambda x: x['timestamp']), 1):
                time_str = issue['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n{i}. [Line {issue['line_number']}] {time_str} - {issue['component']}")
                print(f"   {issue['message']}")
    
    def show_task_failures(self):
        """Show all task failures"""
        print("\n" + "="*60)
        print("TASK FAILURES")
        print("="*60)
        
        if not self.task_failures:
            print("\nNo task failures found in the log.")
            return
            
        for i, failure in enumerate(sorted(self.task_failures, key=lambda x: x['timestamp']), 1):
            time_str = failure['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{i}. [Line {failure['line_number']}] {time_str} - {failure['component']}")
            print(f"   {failure['message']}")
    
    def format_duration(self, seconds):
        """Format a duration in seconds to a human-readable string"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or not parts:
            parts.append(f"{int(seconds)} second{'s' if seconds != 1 else ''}")
            
        return ", ".join(parts)
    
    def show_runtime_analysis(self):
        """Show runtime analysis"""
        print("\n" + "="*60)
        print("RUNTIME ANALYSIS")
        print("="*60)
        
        if not self.first_timestamp or not self.last_timestamp:
            print("\nCould not determine runtime from log data.")
            return
        
        # Overall runtime
        total_duration = (self.last_timestamp - self.first_timestamp).total_seconds()
        print(f"\nOverall Log Duration: {self.format_duration(total_duration)}")
        print(f"First Log Entry: {self.first_timestamp}")
        print(f"Last Log Entry: {self.last_timestamp}")
        
        # Session analysis
        if self.sessions:
            print(f"\nDetected {len(self.sessions)} client sessions:")
            
            for i, session in enumerate(self.sessions, 1):
                print(f"\nSession {i}:")
                print(f"  Started: {session['start_time']}")
                
                if session['end_time']:
                    duration = (session['end_time'] - session['start_time']).total_seconds()
                    print(f"  Ended: {session['end_time']}")
                    print(f"  Duration: {self.format_duration(duration)}")
                else:
                    print(f"  Ended: Unknown (possibly abnormal termination)")
                    print(f"  Duration: Unknown")
                
                print(f"  Connected Successfully: {'Yes' if session.get('connection_established') else 'No'}")
                print(f"  Errors During Session: {session['errors']}")
                print(f"  Warnings During Session: {session['warnings']}")
        else:
            print("\nNo distinct client sessions detected in the log.")
            
        # Connection stability
        if self.connection_attempts > 0:
            print(f"\nConnection Stability:")
            print(f"  Average connections per hour: {self.connection_attempts/(total_duration/3600):.2f}")
            print(f"  Connection failures per hour: {self.connection_failures/(total_duration/3600):.2f}")
            if self.successful_connections > 0:
                avg_duration = total_duration / self.successful_connections
                print(f"  Average connection duration: {self.format_duration(avg_duration)}")
    
    def search_issues(self, query):
        """Search for specific issues in errors and connection problems"""
        print("\n" + "="*60)
        print(f"SEARCH RESULTS FOR: {query}")
        print("="*60)
        
        query = query.lower()
        results = []
        
        # Search in errors
        for error in self.errors:
            if query in error['message'].lower() or query in error['component'].lower():
                results.append({
                    'type': 'ERROR',
                    'timestamp': error['timestamp'],
                    'component': error['component'],
                    'message': error['message'],
                    'line_number': error['line_number']
                })
        
        # Search in warnings
        for warning in self.warnings:
            if query in warning['message'].lower() or query in warning['component'].lower():
                results.append({
                    'type': 'WARNING',
                    'timestamp': warning['timestamp'],
                    'component': warning['component'],
                    'message': warning['message'],
                    'line_number': warning['line_number']
                })
        
        # Search in connection issues
        for issue in self.connection_issues:
            if query in issue['message'].lower() or query in issue['component'].lower():
                results.append({
                    'type': issue['level'],
                    'timestamp': issue['timestamp'],
                    'component': issue['component'],
                    'message': issue['message'],
                    'line_number': issue['line_number']
                })
        
        # Display results
        if not results:
            print(f"\nNo results found for '{query}'")
            return
        
        print(f"\nFound {len(results)} matches:")
        
        # Sort results by timestamp
        for i, result in enumerate(sorted(results, key=lambda x: x['timestamp']), 1):
            time_str = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{i}. [{result['type']}] [Line {result['line_number']}] {time_str} - {result['component']}")
            print(f"   {result['message']}")
    
    def show_menu(self):
        """Display an interactive menu for the user"""
        while True:
            print("\n" + "="*60)
            print("FELIX TRACKING CLIENT LOG ANALYZER")
            print("="*60)
            print("\nWhat would you like to see?")
            print("1. Quick Summary")
            print("2. All Errors (chronological)")
            print("3. Error Categories (grouped)")
            print("4. Connection Issues and Reconnects")
            print("5. Task Failures")
            print("6. Runtime Analysis")  # New option
            print("7. Search for Specific Issues")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-7): ")
            
            if choice == '0':
                print("\nExiting log analyzer. Goodbye!")
                break
            elif choice == '1':
                self.show_quick_summary()
            elif choice == '2':
                self.show_all_errors()
            elif choice == '3':
                self.show_error_categories()
            elif choice == '4':
                self.show_connection_issues()
            elif choice == '5':
                self.show_task_failures()
            elif choice == '6':
                self.show_runtime_analysis()  # New method
            elif choice == '7':
                query = input("\nEnter search term: ")
                self.search_issues(query)
            else:
                print("\nInvalid choice. Please try again.")
            
            input("\nPress Enter to continue...")

def main():
    """Main function to run the simplified log analyzer"""
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the log file.")
        print("Usage: python simple_log_analyzer.py \"C:\\path\\to\\your\\log_file.log\"")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    try:
        analyzer = SimpleLogAnalyzer(log_file)
        analyzer.parse_log()
        analyzer.show_menu()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during log analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()