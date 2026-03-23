/**
 * SyncSettingsScreen -- Google Drive sync management.
 *
 * Google account sign-in/out, WiFi-only toggle, auto-resolve toggle,
 * manual sync, full backup upload, restore from Drive, conflict review,
 * and custom Drive folder option.
 */

import React, {useState, useEffect, useMemo} from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ScrollView,
  Pressable,
  Switch,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useSyncStore } from '../store/useSyncStore';
import { signInToGoogle, signOutGoogle, addDriveFileScope } from '../services/sync/driveAuth';
import { triggerManualSync } from '../services/sync/syncScheduler';
import { performFullBackup } from '../services/backup/backupService';
import { uploadFullBackup } from '../services/sync/driveSync';
import { discoverRemoteBackups, restoreFromDrive } from '../services/sync/restoreService';
import {
  saveFtpCredentials,
  loadFtpCredentials,
  clearFtpCredentials,
  testFtpConnection,
} from '../services/sync/ftpClient';
import type { FtpCredentials } from '../services/sync/ftpClient';
import ConflictResolverModal from '../components/sync/ConflictResolverModal';
import { Paths } from 'expo-file-system';
import { useTheme } from '../theme/ThemeProvider';
import type { ThemeColors } from '../theme/colors';

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function SyncSettingsScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  const {
    signedIn,
    userEmail,
    lastSyncAt,
    syncStatus,
    pendingConflicts,
    wifiOnly,
    autoResolve,
    ftpEnabled,
    ftpHost,
    lastFtpSyncAt,
    ftpSyncStatus,
    setSignedIn,
    setWifiOnly,
    setAutoResolve,
    setFtpEnabled,
    setFtpHost,
    clearSync,
  } = useSyncStore();

  const [syncing, setSyncing] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [conflictModalVisible, setConflictModalVisible] = useState(false);
  const [customFolder, setCustomFolder] = useState(false);

  // FTP form state
  const [ftpFormHost, setFtpFormHost] = useState('');
  const [ftpFormPort, setFtpFormPort] = useState('21');
  const [ftpFormUser, setFtpFormUser] = useState('');
  const [ftpFormPass, setFtpFormPass] = useState('');
  const [ftpFormPath, setFtpFormPath] = useState('/');
  const [testingFtp, setTestingFtp] = useState(false);
  const [savingFtp, setSavingFtp] = useState(false);

  // Load FTP credentials on mount if FTP is enabled
  useEffect(() => {
    if (ftpEnabled) {
      loadFtpCredentials().then((creds) => {
        if (creds) {
          setFtpFormHost(creds.host);
          setFtpFormPort(String(creds.port));
          setFtpFormUser(creds.username);
          setFtpFormPass(creds.password);
          setFtpFormPath(creds.remotePath);
        }
      });
    }
  }, [ftpEnabled]);

  // -----------------------------------------------------------------------
  // Google account
  // -----------------------------------------------------------------------

  async function handleSignIn() {
    try {
      const result = await signInToGoogle();
      const email =
        (result as { data?: { user?: { email?: string } } })?.data?.user?.email ??
        (result as { user?: { email?: string } })?.user?.email ??
        null;
      setSignedIn(true, email);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Sign-in failed';
      Alert.alert('Sign-in Error', msg);
    }
  }

  async function handleSignOut() {
    try {
      await signOutGoogle();
      clearSync();
    } catch {
      Alert.alert('Error', 'Failed to sign out.');
    }
  }

  // -----------------------------------------------------------------------
  // Manual actions
  // -----------------------------------------------------------------------

  async function handleSyncNow() {
    setSyncing(true);
    try {
      await triggerManualSync();
      Alert.alert('Sync Complete', 'Your data has been synced to Google Drive.');
    } catch {
      Alert.alert('Sync Failed', 'Unable to sync. Please try again.');
    } finally {
      setSyncing(false);
    }
  }

  async function handleUploadFull() {
    setUploading(true);
    try {
      const result = await performFullBackup();
      const localPath = `${Paths.document.uri}/backups/${result.filename}`;
      await uploadFullBackup(result.filename, localPath);
      Alert.alert('Upload Complete', `Full backup uploaded: ${result.filename}`);
    } catch {
      Alert.alert('Upload Failed', 'Unable to upload full backup.');
    } finally {
      setUploading(false);
    }
  }

  async function handleRestore() {
    setRestoring(true);
    try {
      const manifest = await discoverRemoteBackups();
      if (!manifest) {
        Alert.alert('No Backups', 'No remote backups found on Google Drive.');
        setRestoring(false);
        return;
      }

      const lastSync = manifest.lastSyncedAt
        ? new Date(manifest.lastSyncedAt).toLocaleString()
        : 'Unknown';

      Alert.alert(
        'Restore from Drive',
        `Found backup from ${lastSync}.\n\nThis will replace your local data. The app will need to restart after restore.`,
        [
          { text: 'Cancel', style: 'cancel', onPress: () => setRestoring(false) },
          {
            text: 'Restore',
            style: 'destructive',
            onPress: async () => {
              try {
                await restoreFromDrive();
                Alert.alert('Restore Complete', 'Data restored. Please restart the app.');
              } catch (err) {
                const msg = err instanceof Error ? err.message : 'Restore failed';
                Alert.alert('Restore Failed', msg);
              } finally {
                setRestoring(false);
              }
            },
          },
        ],
      );
    } catch {
      Alert.alert('Error', 'Failed to check for remote backups.');
      setRestoring(false);
    }
  }

  // -----------------------------------------------------------------------
  // FTP actions
  // -----------------------------------------------------------------------

  async function handleFtpToggle(enabled: boolean) {
    setFtpEnabled(enabled);
    if (!enabled) {
      await clearFtpCredentials();
      setFtpHost(null);
      setFtpFormHost('');
      setFtpFormPort('21');
      setFtpFormUser('');
      setFtpFormPass('');
      setFtpFormPath('/');
    }
  }

  async function handleSaveFtp() {
    const port = parseInt(ftpFormPort, 10);
    if (!ftpFormHost || isNaN(port) || !ftpFormUser) {
      Alert.alert('Missing Fields', 'Please fill in host, port, and username.');
      return;
    }
    setSavingFtp(true);
    try {
      const creds: FtpCredentials = {
        host: ftpFormHost,
        port,
        username: ftpFormUser,
        password: ftpFormPass,
        remotePath: ftpFormPath || '/',
      };
      await saveFtpCredentials(creds);
      setFtpHost(ftpFormHost);
      Alert.alert('Saved', 'FTP credentials saved securely.');
    } catch {
      Alert.alert('Error', 'Failed to save FTP credentials.');
    } finally {
      setSavingFtp(false);
    }
  }

  async function handleTestFtp() {
    setTestingFtp(true);
    try {
      // Save first so testFtpConnection reads the latest values
      const port = parseInt(ftpFormPort, 10);
      if (!ftpFormHost || isNaN(port) || !ftpFormUser) {
        Alert.alert('Missing Fields', 'Please fill in host, port, and username.');
        setTestingFtp(false);
        return;
      }
      await saveFtpCredentials({
        host: ftpFormHost,
        port,
        username: ftpFormUser,
        password: ftpFormPass,
        remotePath: ftpFormPath || '/',
      });
      setFtpHost(ftpFormHost);

      const ok = await testFtpConnection();
      if (ok) {
        Alert.alert('Success', 'FTP connection successful!');
      } else {
        Alert.alert('Failed', 'Could not connect to FTP server. Check your credentials.');
      }
    } catch {
      Alert.alert('Error', 'FTP connection test failed.');
    } finally {
      setTestingFtp(false);
    }
  }

  // -----------------------------------------------------------------------
  // Custom folder scope escalation
  // -----------------------------------------------------------------------

  async function handleCustomFolderToggle(enabled: boolean) {
    if (enabled) {
      try {
        await addDriveFileScope();
        setCustomFolder(true);
      } catch {
        Alert.alert('Error', 'Failed to request Drive file access.');
      }
    } else {
      setCustomFolder(false);
    }
  }

  // -----------------------------------------------------------------------
  // Status helpers
  // -----------------------------------------------------------------------

  function statusColor(): string {
    switch (syncStatus) {
      case 'syncing':
        return colors.accent.amber;
      case 'error':
        return colors.accent.red;
      case 'conflict':
        return colors.accent.amber;
      default:
        return colors.accent.green;
    }
  }

  function statusLabel(): string {
    switch (syncStatus) {
      case 'syncing':
        return 'Syncing...';
      case 'error':
        return 'Error';
      case 'conflict':
        return 'Conflicts';
      default:
        return 'Up to date';
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Google Drive Sync</Text>

      {/* Google Account */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Google Account</Text>
        {signedIn ? (
          <>
            <View style={styles.row}>
              <Ionicons name="person-circle-outline" size={20} color={colors.accent.green} />
              <Text style={styles.emailText}>{userEmail ?? 'Signed in'}</Text>
            </View>
            <Pressable style={styles.actionBtn} onPress={handleSignOut}>
              <Ionicons name="log-out-outline" size={18} color={colors.accent.red} />
              <Text style={[styles.actionBtnText, { color: colors.accent.red }]}>Sign Out</Text>
            </Pressable>
          </>
        ) : (
          <Pressable style={[styles.actionBtn, styles.primaryBtn]} onPress={handleSignIn}>
            <Ionicons name="logo-google" size={18} color={colors.text.inverse} />
            <Text style={[styles.actionBtnText, { color: colors.text.inverse }]}>Sign in with Google</Text>
          </Pressable>
        )}
      </View>

      {/* Status */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Sync Status</Text>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>Status</Text>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
            <View style={[styles.statusDot, { backgroundColor: statusColor() }]} />
            <Text style={styles.rowValue}>{statusLabel()}</Text>
          </View>
        </View>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>Last synced</Text>
          <Text style={styles.rowValue}>
            {lastSyncAt ? relativeTime(lastSyncAt) : 'Never'}
          </Text>
        </View>
      </View>

      {/* Sync Preferences */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Sync Preferences</Text>
        <View style={styles.toggleRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>WiFi only</Text>
            <Text style={styles.hint}>Only sync when connected to WiFi</Text>
          </View>
          <Switch
            value={wifiOnly}
            onValueChange={setWifiOnly}
            trackColor={{ true: colors.accent.green }}
          />
        </View>
        <View style={styles.toggleRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>Auto-resolve conflicts</Text>
            <Text style={styles.hint}>
              When enabled, conflicts are automatically resolved using the most recent change
            </Text>
          </View>
          <Switch
            value={autoResolve}
            onValueChange={setAutoResolve}
            trackColor={{ true: colors.accent.green }}
          />
        </View>
      </View>

      {/* Manual Actions */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Manual Actions</Text>
        {syncing || uploading || restoring ? (
          <ActivityIndicator size="small" color={colors.accent.green} style={{ paddingVertical: 16 }} />
        ) : (
          <>
            <Pressable style={styles.actionBtn} onPress={handleSyncNow} disabled={!signedIn}>
              <Ionicons name="sync-outline" size={18} color={signedIn ? colors.accent.green : colors.border.default} />
              <Text style={[styles.actionBtnText, !signedIn && { color: colors.border.default }]}>
                Sync Now
              </Text>
            </Pressable>
            <Pressable style={styles.actionBtn} onPress={handleUploadFull} disabled={!signedIn}>
              <Ionicons name="cloud-upload-outline" size={18} color={signedIn ? colors.accent.blue : colors.border.default} />
              <Text style={[styles.actionBtnText, { color: signedIn ? colors.accent.blue : colors.border.default }]}>
                Upload Full Backup
              </Text>
            </Pressable>
            <Pressable style={styles.actionBtn} onPress={handleRestore} disabled={!signedIn}>
              <Ionicons name="cloud-download-outline" size={18} color={signedIn ? colors.accent.purple : colors.border.default} />
              <Text style={[styles.actionBtnText, { color: signedIn ? colors.accent.purple : colors.border.default }]}>
                Restore from Drive
              </Text>
            </Pressable>
          </>
        )}
      </View>

      {/* Conflict Review */}
      {pendingConflicts.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Conflicts</Text>
          <View style={styles.row}>
            <View style={styles.conflictBadge}>
              <Text style={styles.conflictBadgeText}>
                {pendingConflicts.length} unresolved conflict{pendingConflicts.length !== 1 ? 's' : ''}
              </Text>
            </View>
          </View>
          <Pressable
            style={[styles.actionBtn, { borderColor: colors.accent.amber }]}
            onPress={() => setConflictModalVisible(true)}
          >
            <Ionicons name="warning-outline" size={18} color={colors.accent.amber} />
            <Text style={[styles.actionBtnText, { color: colors.accent.amber }]}>Review Conflicts</Text>
          </Pressable>
        </View>
      )}

      {/* Custom Folder */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Advanced</Text>
        <View style={styles.toggleRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>Use custom Drive folder</Text>
            <Text style={styles.hint}>
              Default: hidden app data folder. Enable to use a visible folder.
            </Text>
          </View>
          <Switch
            value={customFolder}
            onValueChange={handleCustomFolderToggle}
            trackColor={{ true: colors.accent.purple }}
            disabled={!signedIn}
          />
        </View>
      </View>

      {/* FTP Backup */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>FTP Backup</Text>
        <View style={styles.toggleRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>Enable FTP backup</Text>
            <Text style={styles.hint}>
              Back up to your own FTP server or NAS
            </Text>
          </View>
          <Switch
            value={ftpEnabled}
            onValueChange={handleFtpToggle}
            trackColor={{ true: colors.accent.green }}
          />
        </View>

        {ftpEnabled && (
          <>
            {/* FTP Status */}
            <View style={styles.row}>
              <Text style={styles.rowLabel}>Status</Text>
              <Text style={styles.rowValue}>
                {ftpSyncStatus === 'syncing'
                  ? 'Syncing...'
                  : ftpSyncStatus === 'error'
                    ? 'Error'
                    : ftpHost
                      ? `Connected to ${ftpHost}`
                      : 'Not configured'}
              </Text>
            </View>
            {lastFtpSyncAt && (
              <View style={styles.row}>
                <Text style={styles.rowLabel}>Last FTP sync</Text>
                <Text style={styles.rowValue}>{relativeTime(lastFtpSyncAt)}</Text>
              </View>
            )}

            {/* Credential form */}
            <TextInput
              style={styles.input}
              placeholder="Host (e.g. ftp.example.com)"
              value={ftpFormHost}
              onChangeText={setFtpFormHost}
              autoCapitalize="none"
              autoCorrect={false}
            />
            <TextInput
              style={styles.input}
              placeholder="Port (default 21)"
              value={ftpFormPort}
              onChangeText={setFtpFormPort}
              keyboardType="number-pad"
            />
            <TextInput
              style={styles.input}
              placeholder="Username"
              value={ftpFormUser}
              onChangeText={setFtpFormUser}
              autoCapitalize="none"
              autoCorrect={false}
            />
            <TextInput
              style={styles.input}
              placeholder="Password"
              value={ftpFormPass}
              onChangeText={setFtpFormPass}
              secureTextEntry
              autoCapitalize="none"
              autoCorrect={false}
            />
            <TextInput
              style={styles.input}
              placeholder="Remote path (default /)"
              value={ftpFormPath}
              onChangeText={setFtpFormPath}
              autoCapitalize="none"
              autoCorrect={false}
            />

            {/* Actions */}
            {testingFtp || savingFtp ? (
              <ActivityIndicator size="small" color={colors.accent.green} style={{ paddingVertical: 16 }} />
            ) : (
              <>
                <Pressable style={styles.actionBtn} onPress={handleTestFtp}>
                  <Ionicons name="flash-outline" size={18} color={colors.accent.amber} />
                  <Text style={[styles.actionBtnText, { color: colors.accent.amber }]}>
                    Test Connection
                  </Text>
                </Pressable>
                <Pressable style={styles.actionBtn} onPress={handleSaveFtp}>
                  <Ionicons name="save-outline" size={18} color={colors.accent.green} />
                  <Text style={styles.actionBtnText}>Save Credentials</Text>
                </Pressable>
              </>
            )}
          </>
        )}
      </View>

      <View style={{ height: 100 }} />

      <ConflictResolverModal
        visible={conflictModalVisible}
        onClose={() => setConflictModalVisible(false)}
      />
    </ScrollView>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background.primary },
  content: { paddingTop: 60, paddingHorizontal: 16 },
  title: { fontSize: 28, fontWeight: '800', color: colors.text.primary, marginBottom: 20 },

  card: {
    backgroundColor: colors.background.elevated,
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 3,
  },
  cardTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary, marginBottom: 12 },

  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    gap: 8,
  },
  rowLabel: { fontSize: 15, color: colors.text.secondary },
  rowValue: { fontSize: 15, color: colors.text.tertiary },

  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.background.surface,
  },

  hint: { fontSize: 12, color: colors.text.tertiary, marginTop: 2 },

  emailText: { fontSize: 15, color: colors.text.secondary, marginLeft: 8 },

  actionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.background.surface,
  },
  actionBtnText: { fontSize: 15, fontWeight: '500', color: colors.accent.green },

  primaryBtn: {
    backgroundColor: '#4285F4',
    borderRadius: 12,
    paddingHorizontal: 16,
    justifyContent: 'center',
    borderBottomWidth: 0,
  },

  statusDot: { width: 8, height: 8, borderRadius: 4 },

  conflictBadge: {
    backgroundColor: '#FEF3C7',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  conflictBadgeText: { fontSize: 13, fontWeight: '600', color: '#92400E' },

  input: {
    borderWidth: 1,
    borderColor: colors.border.subtle,
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: 15,
    color: colors.text.primary,
    backgroundColor: colors.background.surface,
    marginTop: 8,
  },
});
}
