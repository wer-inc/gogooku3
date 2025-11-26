import { useState } from 'react'
import { useUsers, useCreateUser, useDeleteUser } from '../hooks/useUsers'
import { Card } from '../components/Card'
import { StatusBadge } from '../components/StatusBadge'

export function Users() {
  const { data: users, isLoading, error } = useUsers()
  const createUser = useCreateUser()
  const deleteUser = useDeleteUser()
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    password: '',
    is_admin: false,
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createUser.mutate(formData, {
      onSuccess: () => {
        setShowForm(false)
        setFormData({ email: '', username: '', password: '', is_admin: false })
      },
    })
  }

  const handleDelete = (userId: number) => {
    if (confirm('Are you sure you want to delete this user?')) {
      deleteUser.mutate(userId)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Loading users...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">Error loading users: {String(error)}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Users</h1>
        <button
          onClick={() => setShowForm(!showForm)}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          {showForm ? 'Cancel' : 'Add User'}
        </button>
      </div>

      {/* Add User Form */}
      {showForm && (
        <Card className="mb-6">
          <h2 className="text-lg font-semibold mb-4">Add New User</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Email
                </label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full border rounded px-3 py-2"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Username
                </label>
                <input
                  type="text"
                  value={formData.username}
                  onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                  className="w-full border rounded px-3 py-2"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Password
                </label>
                <input
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="w-full border rounded px-3 py-2"
                  required
                />
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="is_admin"
                  checked={formData.is_admin}
                  onChange={(e) => setFormData({ ...formData, is_admin: e.target.checked })}
                  className="mr-2"
                />
                <label htmlFor="is_admin" className="text-sm font-medium text-gray-700">
                  Admin User
                </label>
              </div>
            </div>
            <button
              type="submit"
              disabled={createUser.isPending}
              className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:bg-gray-400"
            >
              {createUser.isPending ? 'Creating...' : 'Create User'}
            </button>
          </form>
        </Card>
      )}

      {/* Users Table */}
      <Card>
        {users && users.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b bg-gray-50">
                  <th className="text-left py-3 px-4">ID</th>
                  <th className="text-left py-3 px-4">Username</th>
                  <th className="text-left py-3 px-4">Email</th>
                  <th className="text-left py-3 px-4">Role</th>
                  <th className="text-left py-3 px-4">Status</th>
                  <th className="text-left py-3 px-4">Created</th>
                  <th className="text-left py-3 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4">{user.id}</td>
                    <td className="py-3 px-4 font-medium">{user.username}</td>
                    <td className="py-3 px-4">{user.email}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 text-xs rounded ${
                        user.is_admin ? 'bg-purple-100 text-purple-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {user.is_admin ? 'Admin' : 'User'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <StatusBadge status={user.is_active ? 'ok' : 'warning'} />
                    </td>
                    <td className="py-3 px-4">
                      {new Date(user.created_at).toLocaleDateString('ja-JP')}
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => handleDelete(user.id)}
                        disabled={deleteUser.isPending}
                        className="text-red-600 hover:underline text-sm"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">No users found</p>
        )}
      </Card>
    </div>
  )
}
