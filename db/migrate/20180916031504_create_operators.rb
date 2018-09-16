class CreateOperators < ActiveRecord::Migration
  def change
    create_table :operators do |t|
      t.string :name
      t.string :username
      t.string :password
      t.string :api_key
      t.string :api_password

      t.timestamps
    end
  end
end
