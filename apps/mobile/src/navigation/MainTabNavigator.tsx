import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { MainTabParamList, RootStackParamList } from '../types';
import { DiaryHomeScreen, ProfileScreen, InsightsScreen } from '../screens';

const Tab = createBottomTabNavigator<MainTabParamList>();

function AddPlaceholder() {
  return <View />;
}

function AddTabButton({ children }: { children: React.ReactNode }) {
  const navigation =
    useNavigation<NativeStackNavigationProp<RootStackParamList>>();

  return (
    <Pressable
      onPress={() => navigation.navigate('AddFood', {})}
      style={styles.fabButton}
    >
      <View style={styles.fabButtonInner}>
        <Text style={styles.fabButtonText}>+</Text>
      </View>
    </Pressable>
  );
}

export default function MainTabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: '#16A34A',
        tabBarInactiveTintColor: '#9CA3AF',
        tabBarStyle: {
          borderTopWidth: StyleSheet.hairlineWidth,
          borderTopColor: '#E5E7EB',
          backgroundColor: '#FFFFFF',
          paddingTop: 4,
        },
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: '600',
        },
      }}
    >
      <Tab.Screen
        name="Today"
        component={DiaryHomeScreen}
        options={{
          tabBarLabel: 'Today',
          tabBarIcon: ({ color, size }) => <Ionicons name="today-outline" size={size} color={color} />,
        }}
      />
      <Tab.Screen
        name="Add"
        component={AddPlaceholder}
        options={{
          tabBarButton: (props) => (
            <AddTabButton>{props.children}</AddTabButton>
          ),
          tabBarLabel: '',
        }}
      />
      <Tab.Screen
        name="Insights"
        component={InsightsScreen}
        options={{
          tabBarLabel: 'Insights',
          tabBarIcon: ({ color, size }) => <Ionicons name="stats-chart-outline" size={size} color={color} />,
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarLabel: 'Profile',
          tabBarIcon: ({ color, size }) => <Ionicons name="person-outline" size={size} color={color} />,
        }}
      />
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  fabButton: {
    top: -12,
    justifyContent: 'center',
    alignItems: 'center',
    flex: 1,
  },
  fabButtonInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#16A34A',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  fabButtonText: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '600',
    lineHeight: 30,
  },
});
