import gradio as gr
from pathlib import Path
import os
import json
from datetime import datetime
import requests
from typing import Dict, List, Optional, Tuple
from enum import Enum

class ChatType(Enum):
    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    GENERAL = "general"

class ProjectManager:
    """Handles project-level operations and persistence"""
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Use ~/.gradio_chat/Projects as default
            base_dir = Path.home() / '.gradio_chat' / 'Projects'
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.projects: Dict[str, 'Project'] = self._load_projects()
        
    def _load_projects(self) -> Dict[str, 'Project']:
        """Load existing projects from disk"""
        projects = {}
        for project_dir in self.base_dir.glob('*'):
            if project_dir.is_dir():
                try:
                    projects[project_dir.name] = Project(project_dir)
                except Exception as e:
                    print(f"Error loading project {project_dir.name}: {e}")
        return projects
    
    def create_project(self, name: str) -> 'Project':
        """Create a new project directory and metadata
        
        Args:
            name: Name of the project. Must be a valid directory name.
            
        Returns:
            Newly created Project instance
            
        Raises:
            ValueError: If project name is invalid or already exists
        """
        # Validate project name
        if not name or not name.strip():
            raise ValueError("Project name cannot be empty")
        
        # Clean project name - replace spaces with underscores and remove special chars
        clean_name = "".join(c for c in name.strip().replace(" ", "_") 
                           if c.isalnum() or c in "_-")
        
        if not clean_name:
            raise ValueError("Project name must contain valid characters")
            
        project_dir = self.base_dir / clean_name
        if project_dir.exists():
            raise ValueError(f"Project {clean_name} already exists")
            
        # Create project directory and structure
        project_dir.mkdir()
        (project_dir / "src").mkdir()  # Source code directory
        (project_dir / "docs").mkdir()  # Documentation directory
        (project_dir / "chat_history").mkdir()  # Chat history directory
        
        # Create initial README
        readme_path = project_dir / "README.md"
        readme_path.write_text(f"# {name}\n\nCreated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        project = Project(project_dir)
        self.projects[clean_name] = project
        return project
    
    def delete_project(self, name: str):
        """Delete a project and all its contents
        
        Args:
            name: Name of the project to delete
            
        Raises:
            ValueError: If project doesn't exist
        """
        if name not in self.projects:
            raise ValueError(f"Project {name} does not exist")
            
        project = self.projects[name]
        try:
            import shutil
            shutil.rmtree(project.project_dir)
            del self.projects[name]
        except Exception as e:
            raise ValueError(f"Error deleting project: {e}")
    
    def get_project_names(self) -> List[str]:
        """Get list of all project names"""
        return list(self.projects.keys())
    
    def get_project(self, name: str) -> Optional['Project']:
        """Get a project by name"""
        return self.projects.get(name)

class Project:
    """Represents a single project with its files and chat history"""
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.metadata_file = project_dir / "project_metadata.json"
        self.chat_history_dir = project_dir / "chat_history"
        self.chat_history_dir.mkdir(exist_ok=True)
        
        self.metadata = self._load_metadata()
        self.file_manager = FileManager(self)
        self.file_manager.load_selection()  # Load saved file selection
        
    def _load_metadata(self) -> dict:
        """Load project metadata from disk
        
        # TODO: Add validation for metadata structure
        # TODO: Add recovery mechanism for corrupted metadata
        """
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {
            "name": self.project_dir.name,
            "created_at": datetime.now().isoformat(),
            "llm_settings": {
                #"model": "qwen2.5-coder:32b",
                "model": "phi3.5",
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            },
            "selected_files": []  # Store selected files for persistence
        }
        
    def get_active_files(self) -> Dict[str, str]:
        """Get currently selected files for context"""
        return self.file_manager.get_selected_files()
    
    def save_chat_history(self, chat_type: ChatType, history: List[Tuple[str, str]]):
        """Save chat history by type
        
        # TODO: Add compression for large chat histories
        # TODO: Add cleanup mechanism for old chat histories
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = self.chat_history_dir / f"{chat_type.value}_{timestamp}.json"
        history_file.write_text(json.dumps(history))
        
    def update_llm_settings(self, settings: dict):
        """Update LLM settings in project metadata"""
        self.metadata["llm_settings"] = {
            **self.metadata.get("llm_settings", {}),
            **settings
        }
        self._save_metadata()
        
    def _save_metadata(self):
        """Save project metadata to disk"""
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))
        
    def get_active_files(self) -> Dict[str, str]:
        """Get currently selected files for context
        
        # TODO: Implement file selection logic
        # TODO: Add file content caching
        """
        active_files = {}
        for file_path in self.project_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                relative_path = file_path.relative_to(self.project_dir)
                active_files[str(relative_path)] = file_path.read_text()
        return active_files

class LLMInterface:
    """Handles communication with Ollama API"""
    def __init__(self, host: str = "127.0.0.1", port: int = 11434, model: str = "phi3.5"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        
    async def generate_code(self, prompt: str, context_files: List[str] = None) -> str:
        """Send request to Ollama API
        
        # TODO: Add retry mechanism for failed requests
        # TODO: Add timeout handling
        # TODO: Add proper stream handling for longer responses
        """
        try:
            # Format the prompt with context if provided
            full_prompt = prompt
            if context_files:
                context_content = "\n\nContext files:\n"
                for file in context_files:
                    context_content += f"\n{file}:\n```\n{context_files[file]}\n```\n"
                full_prompt = context_content + "\n\n" + prompt
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,  # Set to True for streaming responses
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
            
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Please ensure Ollama is running."
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
            
    async def stream_generate(self, prompt: str, context_files: List[str] = None):
        """Stream response from Ollama API
        
        # TODO: Add retry mechanism for failed requests
        # TODO: Add timeout handling
        """
        try:
            # Format prompt similar to generate_code
            full_prompt = prompt
            if context_files:
                context_content = "\n\nContext files:\n"
                for file in context_files:
                    context_content += f"\n{file}:\n```\n{context_files[file]}\n```\n"
                full_prompt = context_content + "\n\n" + prompt
                
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                    }
                },
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    yield json_response["response"]
                    
        except Exception as e:
            yield f"Error streaming response: {str(e)}"

from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import json

class FileManager:
    """Enhanced file management with nested directory support and file selection"""
    MAX_TOTAL_SIZE = 1024 * 1024 * 10  # 10MB limit for total selected files
    ALLOWED_EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.md', '.txt', '.json'}
    
    def __init__(self, project: Project):
        self.project = project
        self._selected_files: Set[str] = set()  # Store as relative paths
        
    def is_valid_file(self, file_path: Path) -> bool:
        """Check if file is valid for selection"""
        try:
            if not file_path.is_file():
                return False
                
            # Check file extension
            if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                return False
                
            # Check file size
            if file_path.stat().st_size > (self.MAX_TOTAL_SIZE / 10):  # Individual file limit
                return False
                
            # Try to read first few bytes to ensure it's text
            with file_path.open('rb') as f:
                try:
                    f.read(1024).decode('utf-8')
                except UnicodeDecodeError:
                    return False
                    
            return True
        except Exception:
            return False
            
    def get_total_selected_size(self) -> int:
        """Get total size of all selected files"""
        total_size = 0
        for file_path in self._selected_files:
            try:
                full_path = self.project.project_dir / file_path
                if full_path.is_file():
                    total_size += full_path.stat().st_size
            except Exception:
                continue
        return total_size
        
    def toggle_file_selection(self, file_path: str, selected: bool = None) -> bool:
        """Toggle or set selection state of a file"""
        if not isinstance(file_path, str):
            return False
            
        try:
            # Check file validity when selecting
            if selected and not self.is_valid_file(self.project.project_dir / file_path):
                print(f"Warning: File {file_path} is not valid for selection")
                return False
                
            # Check total size when adding files
            if selected and file_path not in self._selected_files:
                new_total = self.get_total_selected_size()
                try:
                    new_total += (self.project.project_dir / file_path).stat().st_size
                    if new_total > self.MAX_TOTAL_SIZE:
                        print(f"Warning: Adding file would exceed size limit of {self.MAX_TOTAL_SIZE/1024/1024}MB")
                        return False
                except Exception:
                    return False
            
            if selected is None:
                if file_path in self._selected_files:
                    self._selected_files.remove(file_path)
                    return False
                else:
                    self._selected_files.add(file_path)
                    return True
            elif selected:
                self._selected_files.add(file_path)
                return True
            else:
                self._selected_files.discard(file_path)
                return False
                
        except Exception as e:
            print(f"Error in toggle_file_selection: {str(e)}")
            return False
            
    def toggle_directory_selection(self, dir_path: str, selected: bool = None) -> List[Tuple[str, bool]]:
        """Toggle or set selection state for all files in a directory"""
        changes = []
        try:
            dir_path = Path(dir_path) if dir_path else self.project.project_dir
            full_dir_path = self.project.project_dir / dir_path
            
            if not full_dir_path.exists():
                return changes
                
            # First pass: check total size
            if selected:
                total_size = self.get_total_selected_size()
                for file_path in full_dir_path.rglob("*"):
                    if file_path.is_file():
                        try:
                            if self.is_valid_file(file_path):
                                total_size += file_path.stat().st_size
                        except Exception:
                            continue
                            
                if total_size > self.MAX_TOTAL_SIZE:
                    print(f"Warning: Selecting all files would exceed size limit of {self.MAX_TOTAL_SIZE/1024/1024}MB")
                    return changes
            
            # Second pass: actually toggle files
            for file_path in full_dir_path.rglob("*"):
                if file_path.is_file():
                    try:
                        relative_path = str(file_path.relative_to(self.project.project_dir))
                        new_state = self.toggle_file_selection(relative_path, selected)
                        changes.append((relative_path, new_state))
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
                        continue
                        
            self.save_selection()
            
        except Exception as e:
            print(f"Error in toggle_directory_selection: {str(e)}")
            
        return changes
        
    def get_selected_files(self) -> Dict[str, str]:
        """Get currently selected files and their contents"""
        selected_files = {}
        for file_path in self._selected_files:
            try:
                full_path = self.project.project_dir / file_path
                if full_path.is_file() and self.is_valid_file(full_path):
                    selected_files[file_path] = full_path.read_text()
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue
        return selected_files
        
    def save_selection(self):
        """Save current file selection to project metadata"""
        if hasattr(self.project, 'metadata'):
            self.project.metadata["selected_files"] = list(self._selected_files)
            self.project._save_metadata()
            
    def load_selection(self):
        """Load file selection from project metadata"""
        if hasattr(self.project, 'metadata'):
            self._selected_files = set(self.project.metadata.get("selected_files", []))


class CodeEditorInterface:
    """Main interface combining all components"""
    def __init__(self):
        self.project_manager = ProjectManager()  # Uses default ~/.gradio_chat/Projects
        self.llm = LLMInterface()
        self.current_project: Optional[Project] = None
        self.current_chat_type = ChatType.CODE_GENERATION
        
    def switch_project(self, project_name: str) -> bool:
        """Switch current project context
        
        Args:
            project_name: Name of the project to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        if project_name in self.project_manager.projects:
            self.current_project = self.project_manager.projects[project_name]
            # Update LLM settings from project
            settings = self.current_project.metadata.get("llm_settings", {})
            self.llm = LLMInterface(
                #model=settings.get("model", "qwen2.5-coder:32b")
                model=settings.get("model", "phi3.5")
            )
            return True
        return False

def create_ui() -> gr.Blocks:
    """Create the Gradio interface with enhanced layout
    
    Layout Structure:
    ----------------
    [Left Column]         [Right Column]
    Chat Interface        [Tabbed Interface]
                         - File Browser Tab
                         - Project Settings Tab
                         - Chat History Tab
    
    # TODO: Add theme customization
    # TODO: Add keyboard shortcuts
    # TODO: Add session persistence
    """
    interface = CodeEditorInterface()
    
    with gr.Blocks() as demo:
        with gr.Row():
            # Left Column - Chat Interface
            with gr.Column(scale=1):
                chat_type = gr.Radio(
                    choices=[t.value for t in ChatType],
                    value=ChatType.CODE_GENERATION.value,
                    label="Chat Type"
                )
                chatbot = gr.Chatbot(height=650)
                with gr.Row():
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your request here...",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", scale=1)

            # Right Column - Tabbed Interface
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Files"):
                        with gr.Row():
                            refresh_btn = gr.Button("Refresh")
                            new_file_btn = gr.Button("New File")
                            select_all_btn = gr.Button("Select All")
                            clear_selection_btn = gr.Button("Clear Selection")
                        
                        # Add text input for new file name
                        new_file_name = gr.Textbox(
                            label="New File Name",
                            placeholder="Enter file name...",
                            visible=False  # Hidden by default
                        )
                        
                        # Use a Dataframe for file selection
                        file_list = gr.Dataframe(
                            headers=["Selected", "Type", "Path"],
                            datatype=["bool", "str", "str"],
                            interactive=True,
                            col_count=(3, "fixed")
                        )
                        
                        def update_file_tree():
                            """Update the file list display"""
                            if not interface.current_project:
                                return []
                            
                            files = []
                            project_dir = interface.current_project.project_dir
                            file_manager = interface.current_project.file_manager
                            
                            def process_node(path: Path, indent: int = 0):
                                try:
                                    relative_path = str(path.relative_to(project_dir))
                                    if path.is_file():
                                        is_selected = relative_path in file_manager._selected_files
                                        files.append([
                                            is_selected,  # Selected (bool)
                                            "📄",        # Type (str)
                                            relative_path # Path (str)
                                        ])
                                    else:  # Directory
                                        if path != project_dir:  # Skip root directory
                                            files.append([
                                                False,  # Directories can't be directly selected
                                                "📁",
                                                f"{relative_path}/"
                                            ])
                                        # Process children
                                        for child in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name)):
                                            if child.name != "project_metadata.json" and child.name != "chat_history":
                                                process_node(child, indent + 1)
                                except Exception as e:
                                    print(f"Error processing {path}: {str(e)}")
                            
                            process_node(project_dir)
                            return files
                            
                        def handle_selection_change(df):
                            """Handle changes in file selection"""
                            if not interface.current_project or df.empty:
                                return df
                            
                            try:
                                rows = df.values.tolist()  # Correct way to iterate over dataframe
                                for row in rows:
                                    if len(row) >= 3:
                                        selected = bool(row[0])  # First column (Selected)
                                        type_icon = str(row[1])  # Second column (Type)
                                        path = str(row[2])       # Third column (Path)
                                        
                                        if type_icon == "📁":  # Directory
                                            path = path.rstrip("/")
                                        
                                        interface.current_project.file_manager.toggle_file_selection(
                                            path, selected
                                        )
                                
                                interface.current_project.file_manager.save_selection()
                                
                            except Exception as e:
                                print(f"Error in handle_selection_change: {str(e)}")
                                
                            return update_file_tree()
 
                        def toggle_new_file_input():
                            """Toggle visibility of new file input"""
                            return gr.update(visible=True)
                        
                        def create_new_file(filename):
                            """Create a new file in the current project"""
                            if not interface.current_project or not filename:
                                return gr.update(visible=False), file_list.update()
                            
                            try:
                                # Clean filename and ensure it has a proper extension
                                clean_name = filename.strip()
                                if not any(clean_name.endswith(ext) for ext in ['.py', '.md', '.txt', '.json']):
                                    clean_name += '.txt'  # Default to .txt if no extension
                                
                                # Create the file in the project's src directory
                                file_path = interface.current_project.project_dir / 'src' / clean_name
                                file_path.parent.mkdir(exist_ok=True)
                                file_path.write_text('')  # Create empty file
                                
                                # Hide the input and refresh file list
                                return gr.update(visible=False, value=""), update_file_tree()
                                
                            except Exception as e:
                                print(f"Error creating file: {str(e)}")
                                return gr.update(visible=True), file_list.update()
                        
                        def select_all():
                            if interface.current_project:
                                interface.current_project.file_manager.toggle_directory_selection(
                                    "", selected=True
                                )
                                interface.current_project.file_manager.save_selection()
                            return update_file_tree()
                            
                        def clear_selection():
                            if interface.current_project:
                                interface.current_project.file_manager.toggle_directory_selection(
                                    "", selected=False
                                )
                                interface.current_project.file_manager.save_selection()
                            return update_file_tree()
                        
                        # Set up event handlers
                        refresh_btn.click(fn=update_file_tree, outputs=[file_list])
                        new_file_btn.click(fn=toggle_new_file_input, outputs=[new_file_name])
                        new_file_name.submit(fn=create_new_file, inputs=[new_file_name], outputs=[new_file_name, file_list])
                        select_all_btn.click(fn=select_all, outputs=[file_list])
                        clear_selection_btn.click(fn=clear_selection, outputs=[file_list])
                        file_list.change(fn=handle_selection_change, inputs=[file_list], outputs=[file_list])

                    # Project Settings Tab
                    with gr.Tab("Project"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                project_name = gr.Textbox(
                                    label="New Project Name",
                                    placeholder="Enter project name..."
                                )
                                current_project_label = gr.Markdown(
                                    "No project selected",
                                    label="Current Project"
                                )
                            with gr.Column(scale=2):
                                create_project_btn = gr.Button("Create Project")
                                delete_project_btn = gr.Button(
                                    "Delete Project",
                                    variant="stop",
                                )
                        
                        # Initialize project list
                        project_list = gr.Dropdown(
                            label="Switch Project",
                            choices=interface.project_manager.get_project_names(),
                            interactive=True,
                            allow_custom_value=False,
                            value=None
                        )
                        
                        gr.Markdown("### LLM Settings")
                        with gr.Row():
                            model_name = gr.Dropdown(
                                choices=["phi3.5", "qwen2.5-coder:32b", "codellama:34b", "llama2", "mistral"],
                                #value="qwen2.5-coder:32b",
                                value="phi3.5",
                                label="Model",
                                interactive=True
                            )
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature"
                            )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.1,
                            label="Top P"
                        )
                        save_settings_btn = gr.Button("Save LLM Settings")
                        
                        # Project management functions
                        def create_project(name):
                            """Create a new project
                            Args:
                                name: Project name (string)
                            """
                            if not name:
                                return {
                                    project_name: "",
                                    current_project_label: "Error: Project name cannot be empty",
                                    project_list: gr.update(choices=interface.project_manager.get_project_names())
                                }
                                
                            try:
                                project = interface.project_manager.create_project(str(name))
                                interface.current_project = project
                                project_names = interface.project_manager.get_project_names()
                                
                                return {
                                    project_name: "",
                                    current_project_label: f"Current Project: {name}",
                                    project_list: gr.update(choices=project_names, value=name)
                                }
                            except ValueError as e:
                                return {
                                    project_name: name,
                                    current_project_label: f"Error: {str(e)}",
                                    project_list: gr.update(choices=interface.project_manager.get_project_names())
                                }
                                
                        def delete_current_project():
                            if interface.current_project:
                                try:
                                    name = interface.current_project.metadata["name"]
                                    interface.project_manager.delete_project(name)
                                    interface.current_project = None
                                    return {
                                        current_project_label: "No project selected",
                                        project_list: interface.project_manager.get_project_names()
                                    }
                                except ValueError as e:
                                    return {
                                        current_project_label: f"Error: {str(e)}",
                                        project_list: interface.project_manager.get_project_names()
                                    }
                                    
                        def switch_project(name):
                            """Handle project switching
                            Args:
                                name: Project name (string or list containing one string)
                            """
                            # Handle case where name comes as a list
                            if isinstance(name, list):
                                name = name[0] if name else None
                            
                            if name:
                                interface.current_project = interface.project_manager.get_project(str(name))
                                if interface.current_project:
                                    settings = interface.current_project.metadata.get("llm_settings", {})
                                    return {
                                        current_project_label: f"Current Project: {name}",
                                        #model_name: settings.get("model", "qwen2.5-coder:32b"),
                                        model_name: settings.get("model", "phi3.5"),
                                        temperature: settings.get("temperature", 0.7),
                                        top_p: settings.get("top_p", 0.9)
                                    }
                            return {
                                current_project_label: "No project selected",
                                #model_name: "qwen2.5-coder:32b",
                                model_name: "phi3.5",
                                temperature: 0.7,
                                top_p: 0.9
                            }
                            
                        def save_llm_settings(model, temp, p):
                            if interface.current_project:
                                interface.current_project.update_llm_settings({
                                    "model": model,
                                    "temperature": temp,
                                    "top_p": p
                                })
                                return gr.update(value=f"Current Project: {interface.current_project.metadata['name']}\nSettings saved!")
                            return gr.update(value="No project selected")
                        
                        # Set up event handlers
                        create_project_btn.click(
                            fn=create_project,
                            inputs=[project_name],
                            outputs=[project_name, current_project_label, project_list]
                        )
                        
                        delete_project_btn.click(
                            fn=delete_current_project,
                            outputs=[current_project_label, project_list]
                        )
                        
                        project_list.change(
                            fn=switch_project,
                            inputs=[project_list],
                            outputs=[current_project_label, model_name, temperature, top_p]
                        )
                        
                        save_settings_btn.click(
                            fn=save_llm_settings,
                            inputs=[model_name, temperature, top_p],
                            outputs=[current_project_label]
                        )
                    
                    # Chat History Tab
                    # Chat History Tab
                    with gr.Tab("History"):
                        with gr.Row():
                            refresh_history_btn = gr.Button("Refresh History")
                            history_dropdown = gr.Dropdown(
                                label="Select Chat History",
                                choices=[],
                                interactive=True
                            )
                        
                        history_display = gr.Chatbot(
                            label="Chat History",
                            height=500
                        )
                        
                        def list_history_files():
                            if not interface.current_project:
                                return [], gr.update()
                            try:
                                history_dir = interface.current_project.chat_history_dir
                                files = list(history_dir.glob("*.json"))
                                # Format the display names
                                choices = []
                                for f in files:
                                    try:
                                        # The filename format is: code_generation_20241118_222122.json
                                        name_parts = f.stem.split('_')  # Split filename into parts
                                        # Handle chat type (might be multiple parts like 'code' 'generation')
                                        chat_type = ' '.join(name_parts[:-2]).title()
                                        # Last two parts are date and time
                                        date_str = name_parts[-2]
                                        time_str = name_parts[-1]
                                        
                                        # Parse date and time
                                        date = datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
                                        display_name = f"{chat_type} - {date.strftime('%Y-%m-%d %H:%M:%S')}"
                                        choices.append((display_name, str(f.name)))
                                    except Exception as e:
                                        print(f"Error parsing filename {f.name}: {e}")
                                        # Add raw filename as fallback
                                        choices.append((str(f.name), str(f.name)))
                                
                                return gr.update(choices=choices), gr.update()
                            except Exception as e:
                                print(f"Error listing history files: {e}")
                                return [], gr.update()
                        
                        def load_history(filename):
                            if not filename or not interface.current_project:
                                return gr.update()
                            try:
                                history_path = interface.current_project.chat_history_dir / filename
                                history_content = json.loads(history_path.read_text())
                                return history_content
                            except Exception as e:
                                print(f"Error loading history: {e}")
                                return []
                        
                        # Event handlers
                        refresh_history_btn.click(
                            fn=list_history_files,
                            outputs=[history_dropdown, history_display]
                        )
                        
                        history_dropdown.change(
                            fn=load_history,
                            inputs=[history_dropdown],
                            outputs=[history_display]
                        ) 
                    #with gr.Tab("History"):
                    #    gr.HTML("""
                    #        <div id="chat-history-root"></div>
                    #        <script type="module">
                    #            import ChatHistoryViewer from './src/ChatHistoryViewer.jsx';
                    #            
                    #            const root = document.getElementById('chat-history-root');
                    #            if (root && !root.hasChildNodes()) {
                    #                ReactDOM.createRoot(root).render(
                    #                    React.createElement(ChatHistoryViewer)
                    #                );
                    #            }
                    #        </script>
                    #    """)

        # Event handlers
        async def handle_chat(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
            """Handle chat messages with streaming support"""
            if not interface.current_project:
                history = history + [("Error", "No project selected")]
                yield "", history
                return

            # Add user message to history immediately
            history = history + [(message, "")]
            yield "", history
            
            # Get current project settings
            settings = interface.current_project.metadata.get("llm_settings", {})
            
            # Start streaming response
            response_parts = []
            try:
                async for part in interface.llm.stream_generate(
                    prompt=message,
                    context_files=interface.current_project.get_active_files()
                ):
                    response_parts.append(part)
                    # Update history with accumulated response
                    current_response = "".join(response_parts)
                    updated_history = history[:-1] + [(message, current_response)]
                    yield "", updated_history
            except Exception as e:
                error_history = history[:-1] + [(message, f"Error: {str(e)}")]
                yield "", error_history
                return
            
            # Save final history
            interface.current_project.save_chat_history(
                interface.current_chat_type,
                updated_history
            )

        submit_btn.click(
            fn=handle_chat,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",  # Makes it accessible from other computers on the network
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public URL
        auth=None,             # Add tuple of ("username", "password") for authentication
        ssl_verify=False       # Disable SSL verification if needed
    )
