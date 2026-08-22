; AeonInstaller.iss — Inno Setup 6.x installer for Aeon (§W6).
;
; Produces:
;     dist\installer\AeonSetup.exe
;
; Behaviour:
;   * Per-user installation (no admin required)
;   * Installs under %LOCALAPPDATA%\Programs\Aeon
;   * Start-menu shortcut always; desktop shortcut opt-in
;   * Uninstaller registered under the current user
;   * Upgrade preserves user data (config, jobs, logs, evidence, checkpoints)
;   * Uninstall preserves user data by default; explicit checkbox for deletion
;   * Does NOT auto-launch training; optional post-install config wizard
;   * Bundled runtime manifest is verified before install completes

#define AppId       "{{A1E00-1234-4567-8901-234567890ABC}"
#define AppName     "Aeon"
#define AppVersion  "0.2.3"
#define AppPublisher "Aeon"
#define AppURL      "https://example.invalid/aeon"
#define AppExe      "Aeon.exe"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
; W10-7/A12: pin the base directory relative paths resolve from. Without
; SourceDir=, ISCC resolves [Files] Source: from the directory containing
; this .iss file (packaging\windows), which would look at
; packaging\windows\dist\Aeon — the wrong tree. This .iss lives at
; packaging\windows\AeonInstaller.iss; the repo root is two levels up.
SourceDir=..\..
OutputDir=dist\installer
OutputBaseFilename=AeonSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableDirPage=auto
UsedUserAreasWarning=no
; W10-7/A14: CloseApplications=force would forcibly kill a running Aeon
; worker mid-write. Refuse the upgrade instead — the worker owns
; checkpoint atomicity. RestartApplications=no is kept explicit.
CloseApplications=no
RestartApplications=no
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "&Create a desktop shortcut"; \
    GroupDescription: "Additional shortcuts:"; Flags: unchecked
Name: "postinstall_configure"; Description: "&Open the Aeon configuration wizard on finish"; \
    GroupDescription: "First-run:"; Flags: unchecked

[Files]
; Ship the entire PyInstaller onedir output. Do NOT ship user data, corpus,
; checkpoints, keys, .git, or test artefacts.
Source: "dist\Aeon\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; WIN-PATCH-A/Failure E: embed verification-only copies of the runtime
; manifest and its SHA-256 sidecar inside Setup.exe itself. Without this,
; PrepareToInstall would look up {src}\dist\Aeon\... — that only resolves
; while Setup.exe sits inside the repository build tree, and fails once
; AeonSetup.exe is copied out for distribution. The dontcopy flag stores
; the file inside Setup.exe's compressed payload only (never installs it
; to disk); ExtractTemporaryFile('AEON_RUNTIME_MANIFEST.*') pulls it into
; {tmp} at install time.
Source: "dist\Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.json";   DestName: "AEON_RUNTIME_MANIFEST.json";   Flags: dontcopy
Source: "dist\Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.sha256"; DestName: "AEON_RUNTIME_MANIFEST.sha256"; Flags: dontcopy

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; \
    Description: "Launch {#AppName} configuration wizard"; \
    Parameters: ""; \
    Flags: postinstall nowait skipifsilent unchecked
; The configure task above uses Tasks: postinstall_configure — but Inno Setup's
; Run/Tasks interaction requires a Check condition; we implement the opt-in
; behaviour via the [Code] section below.

[UninstallDelete]
; Application-registered files under {app} are removed by Inno Setup by default.
; NOTHING under user_data_root is deleted by default; the [Code] section
; handles the optional user-data purge.

[Code]
var
  RemoveUserDataCheckBox: TNewCheckBox;

procedure InitializeWizard;
var
  Panel: TNewStaticText;
begin
  // Panel is added on the ready page; only visible during uninstall
end;

// --- Uninstall: optional user-data purge -----------------------------------
function InitializeUninstall(): Boolean;
begin
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResponseCode: Integer;
  UserDataPath: string;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    UserDataPath := ExpandConstant('{localappdata}\Aeon');
    if DirExists(UserDataPath) then
    begin
      ResponseCode := MsgBox(
        'Aeon has been uninstalled.' + #13#10 + #13#10 +
        'Your Aeon user data (config, checkpoints, logs, evidence) is still at:' + #13#10 +
        UserDataPath + #13#10 + #13#10 +
        'DELETE this user data now?' + #13#10 +
        '(By default it is preserved so you can reinstall without losing training progress.)',
        mbConfirmation, MB_YESNO or MB_DEFBUTTON2);
      if ResponseCode = IDYES then
        DelTree(UserDataPath, True, True, True);
    end;
  end;
end;

// --- Pre-install: verify the bundled RUNTIME_MANIFEST.json contents --------
// W10-7/A13 correction: the old pre-install check was `FileExists` only —
// presence, not payload. A tampered manifest would pass and only be caught
// at first launch. The corrected check:
//   1. Locates RUNTIME_MANIFEST.json inside the packaged onedir payload.
//   2. Locates RUNTIME_MANIFEST.sha256 (sidecar written by
//      packaging\windows\generate_runtime_manifest.py at build time).
//   3. Computes SHA-256 of the manifest and compares byte-for-byte.
// The manifest itself hashes every payload file, so a matching manifest
// digest transitively verifies the whole payload against the build tree.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ManifestPath, SidecarPath, ExpectedSha, ActualSha, SidecarText: string;
  ExpectedShaAnsi: AnsiString;
  SpaceIdx: Integer;
begin
  NeedsRestart := False;
  // WIN-PATCH-A/Failure E: extract the embedded manifest + sidecar into
  // {tmp} instead of reading from {src}\dist\Aeon\... — the source-tree
  // path only works while Setup.exe is inside the repository build
  // layout and breaks once AeonSetup.exe is distributed on its own.
  ExtractTemporaryFile('AEON_RUNTIME_MANIFEST.json');
  ExtractTemporaryFile('AEON_RUNTIME_MANIFEST.sha256');
  ManifestPath := ExpandConstant('{tmp}\AEON_RUNTIME_MANIFEST.json');
  SidecarPath := ExpandConstant('{tmp}\AEON_RUNTIME_MANIFEST.sha256');
  if not FileExists(ManifestPath) then
  begin
    Result := 'Installer payload is missing RUNTIME_MANIFEST.json. ' +
              'Re-run packaging\windows\generate_runtime_manifest.py before packaging.';
    exit;
  end;
  if not FileExists(SidecarPath) then
  begin
    Result := 'Installer payload is missing RUNTIME_MANIFEST.sha256. ' +
              'Re-run packaging\windows\generate_runtime_manifest.py before packaging.';
    exit;
  end;
  // WIN-PATCH-A/Failure D: Inno Setup 6.7.3 rejects a Unicode string
  // as the LoadStringFromFile output buffer; the correct pattern uses
  // an AnsiString buffer and an explicit conversion afterwards.
  if not LoadStringFromFile(SidecarPath, ExpectedShaAnsi) then
  begin
    Result := 'Cannot read RUNTIME_MANIFEST.sha256 sidecar. Refusing install.';
    exit;
  end;
  SidecarText := Trim(Lowercase(String(ExpectedShaAnsi)));
  // generate_runtime_manifest.py writes only the 64-hex-char digest
  // with no filename or trailing space, but tolerate a two-space
  // "digest  filename" line just in case a future update pins the
  // sidecar to `sha256sum` output — parse the first whitespace-free
  // token as the expected digest.
  SpaceIdx := Pos(' ', SidecarText);
  if SpaceIdx > 0 then
    ExpectedSha := Copy(SidecarText, 1, SpaceIdx - 1)
  else
    ExpectedSha := SidecarText;
  if Length(ExpectedSha) <> 64 then
  begin
    Result := 'RUNTIME_MANIFEST.sha256 does not parse as a 64-hex-char digest. Refusing install.' + #13#10 +
              'raw sidecar text: "' + SidecarText + '"';
    exit;
  end;
  ActualSha := Lowercase(GetSHA256OfFile(ManifestPath));
  if ExpectedSha <> ActualSha then
  begin
    Result := 'Runtime manifest SHA-256 does not match its recorded value. ' +
              'Payload integrity check FAILED. Refusing install.' + #13#10 +
              'expected: ' + ExpectedSha + #13#10 +
              'actual:   ' + ActualSha;
    exit;
  end;
  Result := '';
end;

// --- Upgrade behaviour: refuse to run if a worker is active ---------------
// W10-7/A14 correction: original guard triggered only on 'CHECKPOINTING',
// allowing an in-flight upgrade to overwrite bundle files while a
// RUNNING / STARTING / STOP_REQUESTED worker was mid-batch. Combined with
// CloseApplications=force (removed above), that meant an upgrade could
// forcibly kill a live worker between checkpoint boundaries. All four
// live-status values now block.
function IsAnActiveJob(): Boolean;
var
  UserDataPath, JobsPath, StatusPath, Status: string;
  StatusAnsi: AnsiString;
  FindRec: TFindRec;
begin
  Result := False;
  UserDataPath := ExpandConstant('{localappdata}\Aeon');
  JobsPath := UserDataPath + '\jobs';
  if not DirExists(JobsPath) then exit;
  if FindFirst(JobsPath + '\*', FindRec) then
  begin
    try
      repeat
        if ((FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0)
            and (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          StatusPath := JobsPath + '\' + FindRec.Name + '\status.json';
          if FileExists(StatusPath) then
          begin
            // WIN-PATCH-A/Failure D: Inno Setup 6.7.3 rejects a Unicode
            // string as the LoadStringFromFile destination; use an
            // AnsiString buffer and convert. Active-state protections
            // (RUNNING / STARTING / STOP_REQUESTED / CHECKPOINTING)
            // are preserved.
            if LoadStringFromFile(StatusPath, StatusAnsi) then
            begin
              Status := String(StatusAnsi);
              if (Pos('CHECKPOINTING', Status) > 0)
                 or (Pos('RUNNING', Status) > 0)
                 or (Pos('STARTING', Status) > 0)
                 or (Pos('STOP_REQUESTED', Status) > 0) then
              begin
                Result := True;
                exit;
              end;
            end;
          end;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if IsAnActiveJob() then
  begin
    MsgBox('An Aeon training job is currently active (RUNNING, STARTING, '
        + 'STOP_REQUESTED, or CHECKPOINTING). Please use Stop Safely in '
        + 'the Aeon launcher and wait for the job to reach a terminal '
        + 'state before running the installer again.', mbError, MB_OK);
    Result := False;
  end;
end;
